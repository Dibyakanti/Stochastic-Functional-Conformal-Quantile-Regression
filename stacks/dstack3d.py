# Copyright 2025 The swirl_dynamics Authors.
# Modification by Dibyakanti Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import MutableSequence, Sequence
from typing import Any, Literal

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from layers.convolutions import ConvLayer, DownsampleConv, ConvBlock
from layers.axial_attention import AxialSelfAttentionBlock
from utils.model_utils import as_3d_ratio, default_init

Array = jax.Array
Initializer = nn.initializers.Initializer
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)

class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling
  (spatial only, temporal dimension is left untouched). Features at different
  resolutions are concatenated into output to use for skip connections by the
  UStack module.
  """

  num_channels: Sequence[int]
  num_res_blocks: Sequence[int]
  downsample_ratio: Sequence[int | tuple[int, int, int]]
  use_spatial_attention: Sequence[bool]
  # use_temporal_attention: Sequence[bool]
  num_input_proj_channels: int = 128
  padding: str = "LATLON"
  dropout_rate: float = 0.0
  use_position_encoding: bool = False
  precision: PrecisionLike = None
  num_heads: int = 8
  normalize_qk: bool = False
  ffn_type: Literal["dense", "swiglu", "geglu"] = "dense"
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, emb: Array, is_training: bool) -> list[Array]:
    # The following should already have been checked at the caller level. Run
    # assserts to verify that it is indeed the case.
    if x.ndim != 5:
      raise ValueError(
          "Only accept 5D input (batch, time, x, y, features)! "
          f"Instead we have x.shape: {x.shape}"
      )
    if x.shape[0] != emb.shape[0]:
      raise ValueError(
          "Batch dimension of x and emb must match! "
          f"x.shape: {x.shape}, emb.shape: {emb.shape}"
      )
    # if len(self.use_spatial_attention) != len(self.use_temporal_attention):
    #   raise ValueError(
    #       "Length of use_spatial_attention and use_temporal_attention must"
    #       " match!"
    #   )
    if len(self.num_channels) != len(self.num_res_blocks):
      raise ValueError("Length of num_channels and num_res_blocks must match!")
    if len(self.downsample_ratio) != len(self.num_res_blocks):
      raise ValueError(
          "Length of downsample_ratio and num_res_blocks must match!"
      )

    # Casting to the specified dtype.
    orig_dtype = x.dtype
    x = x.astype(self.dtype)
    emb = emb.astype(self.dtype)

    # Logic.
    spatial_dims = np.asarray(x.shape[2:-1])
    nt = x.shape[1]
    skips = []

    # Spatial projection to tokens.
    h = ConvLayer(
        features=self.num_input_proj_channels,
        kernel_size=(3, 3, 3),
        padding=self.padding,
        kernel_init=default_init(1.0),
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="conv3d_in",
    )(x)
    skips.append(h)

    # Downsampling.
    for level, channel in enumerate(self.num_channels):
      ratios = as_3d_ratio(self.downsample_ratio[level])
      h = DownsampleConv(
          features=channel,
          ratios=ratios,
          kernel_init=default_init(1.0),
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"{nt}xres{'x'.join(spatial_dims.astype(str))}.downsample",
      )(h)
      spatial_dims = spatial_dims // np.asarray(ratios[-len(spatial_dims):])
      dims_str = spatial_dims.astype(str)

      # First part of the processing. The rest will be done in the UStack.
      for block_id in range(self.num_res_blocks[level]):
        h = ConvBlock(
            out_channels=channel,
            kernel_size=(3, 3, 3),
            padding=self.padding,
            dropout=self.dropout_rate,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{nt}xres{'x'.join(dims_str)}.dblock{block_id}",
        )(h, emb, is_training=is_training)
        # Applies attention in space/time if needed.
        if (
            self.use_spatial_attention[level]
            # or self.use_temporal_attention[level]
        ):
          attn_axes = []
          if self.use_spatial_attention[level]:
            attn_axes.extend([1, 2, 3])
          # if self.use_temporal_attention[level]:
          #   attn_axes.append(1)

          h = AxialSelfAttentionBlock(
              attention_axes=attn_axes,
              add_position_embedding=self.use_position_encoding,
              num_heads=self.num_heads,
              precision=self.precision,
              dtype=self.dtype,
              param_dtype=self.param_dtype,
              normalize_qk=self.normalize_qk,
              ffn_type=self.ffn_type,
              name=f"{nt}xres{'x'.join(dims_str)}.dblock{block_id}.attn",
          )(h, is_training)

        h = h.astype(orig_dtype)
        skips.append(h)

    return skips