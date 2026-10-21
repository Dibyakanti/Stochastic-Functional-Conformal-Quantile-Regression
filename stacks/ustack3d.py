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

from layers.convolutions import ConvLayer, ConvBlock
from layers.axial_attention import AxialSelfAttentionBlock
from layers.residual import CombineResidualWithSkip
from layers.upsample import channel_to_space
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


class UStack(nn.Module):
  """Upsampling Stack.

  Takes in features at intermediate resolutions from the downsampling stack
  as well as its final output, and applies upsampling with convolutional blocks
  and combines together with skip connections in typical UNet style.
  """

  num_channels: Sequence[int]
  num_res_blocks: Sequence[int]
  upsample_ratio: Sequence[int | tuple[int, int, int]]
  use_spatial_attention: Sequence[bool]
  # use_temporal_attention: Sequence[bool]
  num_output_proj_channels: int = 128
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_position_encoding: bool = False
  num_heads: int = 8
  precision: PrecisionLike = None
  normalize_qk: bool = False
  ffn_type: Literal["dense", "swiglu", "geglu"] = "dense"
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: Array,
      emb: Array,
      skips: MutableSequence[Array],
      is_training: bool,
  ) -> Array:
    # The following should already have been checked at the caller level.
    # We add an extra layer of checks here.
    if x.ndim != 5:
      raise ValueError(
          "Only accept 5D input (batch, x, y, z, features)! "
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
    if len(self.upsample_ratio) != len(self.num_res_blocks):
      raise ValueError(
          "Length of upsample_ratio and num_res_blocks must match!"
      )

    # Casting to the specified dtype.
    orig_dtype = x.dtype
    x = x.astype(self.dtype)
    emb = emb.astype(self.dtype)

    # Logic.
    spatial_dims = np.asarray(x.shape[-3:-1])
    dims_str = spatial_dims.astype(str)
    nt = x.shape[1]
    h = x

    # Takes the sking and it processes them if needed.
    for level, channel in enumerate(self.num_channels):
      for block_id in range(self.num_res_blocks[level]):
        last_skip = skips.pop().astype(self.dtype)
        h = CombineResidualWithSkip(
            project_skip=h.shape[-1] != last_skip.shape[-1],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(residual=h, skip=last_skip)
        h = ConvBlock(
            out_channels=channel,
            # This will treat the time dimension as an extra batch dimension.
            kernel_size=(3, 3, 3),
            padding=self.padding,
            dropout=self.dropout_rate,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{nt}xres{'x'.join(dims_str)}.ublock{block_id}",
        )(h, emb, is_training=is_training)
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
              name=f"{nt}xres{'x'.join(dims_str)}.ublock{block_id}.attn",
          )(h, is_training)

      # Upsampling.
      up_ratio = as_3d_ratio(self.upsample_ratio[level])
      h = ConvLayer(
          features=int(np.prod(up_ratio)) * channel,
          kernel_size=(3, 3, 3),
          padding=self.padding,
          kernel_init=default_init(1.0),
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"{nt}xres{'x'.join(dims_str)}.conv3d_preupsample",
      )(h)
      h = channel_to_space(inputs=h, block_shape=up_ratio)
      spatial_dims = spatial_dims * np.asarray(up_ratio[-len(spatial_dims):])
      dims_str = spatial_dims.astype(str)

    h = CombineResidualWithSkip(
        project_skip=h.shape[-1] != skips[-1].shape[-1],
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(residual=h, skip=skips.pop())
    h = ConvLayer(
        features=self.num_output_proj_channels,
        kernel_size=(3, 3, 3),
        padding=self.padding,
        kernel_init=default_init(1.0),
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="conv3d_out",
    )(h)
    return h.astype(orig_dtype)