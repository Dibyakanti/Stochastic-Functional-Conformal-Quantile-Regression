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

"""3D U-Net denoiser models.

Intended for inputs with dimensions (batch, x, y, z, channels). The U-Net
stacks successively apply 2D downsampling/upsampling in space only. At each
resolution, an axial attention block (involving space and/or time) is applied.
"""

from collections.abc import Sequence
from typing import Any, Literal

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from layers.convolutions import ConvLayer
from embeddings.fourier_emb import FourierEmbedding
from layers.resize import FilteredResize, InterpConvMerge
from stacks.dstack3d import DStack
from stacks.ustack3d import UStack
from utils.model_utils import default_init

Array = jax.Array
Initializer = nn.initializers.Initializer
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)

def _maybe_broadcast_to_list(
    source: bool | Sequence[bool], reference: Sequence[Any]
) -> list[bool]:
  """Broadcasts to a list with the same length if applicable."""
  if isinstance(source, bool):
    return [source] * len(reference)
  else:
    if len(source) != len(reference):
      raise ValueError(f"{source} must have the same length as {reference}!")
    return list(source)


class UNet3d(nn.Module):
  """UNet model for 3D time-space input.

  This model processes 3D spatiotemporal data using a UNet architecture. It
  progressively downsamples the input for efficient feature extraction at
  multiple scales. Features are extracted using 2D spatial convolutions along
  with spatial and/or temporal axial attention blocks. Upsampling and
  combination of features across scales produce an output with the same shape as
  the input.

  Attributes:
    out_channels: Number of output channels (should match the input).
    resize_to_shape: Optional input resizing shape. Facilitates greater
      downsampling flexibility. Output is resized to the original input shape.
    num_channels: Number of feature channels in intermediate convolutions.
    downsample_ratio: Spatial downsampling ratio per resolution (must evenly
      divide spatial dimensions).
    num_blocks: Number of residual convolution blocks per resolution.
    noise_embed_dim: Embedding dimensions for noise levels.
    input_proj_channels: Number of input projection channels.
    output_proj_channels: Number of output projection channels.
    padding: 2D padding type for spatial convolutions.
    dropout_rate: Dropout rate between convolution layers.
    use_spatial_attention: Whether to enable axial attention in spatial
      directions at each resolution.
    # use_temporal_attention: Whether to enable axial attention in the temporal
      direction at each resolution.
    use_position_encoding: Whether to add position encoding before axial
      attention.
    num_heads: Number of attention heads.
    normalize_qk: Whether to apply normalization to the Q and K matrices of
      attention layers.
    # cond_resize_method: Resize method for channel-wise conditioning.
    # cond_embed_dim: Embedding dimension for channel-wise conditioning.
    ffn_type: Type of feed-forward network for attention blocks.
    precision: Precision for attention blocks.
    dtype: Data type of the input/output.
    param_dtype: Parameter data type for the weights of the model.
  """
  out_channels: int
  time_cond: bool = True
  resize_to_shape: Sequence[int] | None = None
  num_channels: Sequence[int] = (128, 256, 256, 256)
  downsample_ratio: Sequence[int] = (2, 2, 2, 2)
  num_blocks: int = 4
  noise_embed_dim: int = 128
  input_proj_channels: int = 128
  output_proj_channels: int = 128
  kernel_size: tuple[int, int, int] = (3, 3, 3)  # (3, 3, 1) = 2D conv on a singleton z axis
  padding: str = "SAME"
  dropout_rate: float = 0.0
  use_spatial_attention: bool | Sequence[bool] = (False, False, False, True)
  # use_temporal_attention: bool | Sequence[bool] = (False, False, False, True)
  use_position_encoding: bool = True
  num_heads: int = 8
  normalize_qk: bool = False
  # cond_resize_method: str = "cubic"
  # cond_embed_dim: int = 128
  ffn_type: Literal["dense", "swiglu", "geglu"] = "dense"
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: Array,
      # sigma: Array,
      time: Array,
      # cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Predicts denoised given noised input and noise level.

    Args:
      x: The model input (i.e. noised sample) with shape `(batch,
        **spatial_dims, channels)`.
      sigma: The noise level, which either shares the same batch dimension as
        `x` or is a scalar (will be broadcasted accordingly).
      cond: The conditional inputs as a dictionary. Currently, only channelwise
        conditioning is supported.
      is_training: A boolean flag that indicates whether the module runs in
        training mode.

    Returns:
      An output array with the same dimension as `x`.
    """
    # if sigma.ndim < 1:
    #   sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    # if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
    #   raise ValueError(
    #       "`sigma` must be 1D and have the same leading (batch) dimension as x"
    #       f" ({x.shape[0]})!"
    #   )
    if self.time_cond:
      if time.ndim < 1:
        time = jnp.broadcast_to(time, (x.shape[0],))

    if time.ndim != 1 or x.shape[0] != time.shape[0]:
      raise ValueError(
          "`time` must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    if x.ndim != 5:
      raise ValueError(
          "Only accept 5D input (batch, x, y, x, features)! x.shape:"
          f" {x.shape}"
      )

    # If the attention are a single boolean, we broadcast it to the number of
    # number of levels in the Unet.
    use_spatial_attn = _maybe_broadcast_to_list(
        source=self.use_spatial_attention, reference=self.num_channels
    )
    # use_temporal_attn = _maybe_broadcast_to_list(
    #     source=self.use_temporal_attention, reference=self.num_channels
    # )

    if len(self.num_channels) != len(self.downsample_ratio):
      raise ValueError(
          f"`num_channels` {self.num_channels} and `downsample_ratio`"
          f" {self.downsample_ratio} must have the same lengths!"
      )

    # Casting to the specified dtype.
    orig_dtype = x.dtype
    x = x.astype(self.dtype)

    # input_size = x.shape[1:-1]
    # # Resizes the input if needed. This steps allows for greater downsampling
    # # flexibility. The output is resized to the original input shape at the end.
    # if self.resize_to_shape is not None:
    #   x = FilteredResize(
    #       output_size=self.resize_to_shape,
    #       kernel_size=(7, 7),
    #       padding=self.padding,
    #       precision=self.precision,
    #       dtype=self.dtype,
    #       param_dtype=self.param_dtype,
    #   )(x)

    # # Merges the conditional inputs to the main input. The conditional inputs
    # # are resized to match the spatial shape of the main input.
    # cond = {} if cond is None else cond
    # x = InterpConvMerge(
    #     embed_dim=self.cond_embed_dim,
    #     resize_method=self.cond_resize_method,
    #     kernel_size=(3, 3),
    #     padding=self.padding,
    #     precision=self.precision,
    #     dtype=self.dtype,
    #     param_dtype=self.param_dtype,
    # )(x, cond)

    # Adds the noise and the time conditioning.
    # if self.time_cond:
    #   emb_sigma = FourierEmbedding(
    #       dims=self.noise_embed_dim,
    #       dtype=self.dtype,
    #       param_dtype=self.param_dtype,
    #   )(sigma)
      
    emb_time = FourierEmbedding(
        dims=self.noise_embed_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(time)
      
    #   emb = jnp.concatenate((emb_sigma, emb_time), axis=-1)
    # else:
    #   emb = FourierEmbedding(
    #       dims=self.noise_embed_dim,
    #       dtype=self.dtype,
    #       param_dtype=self.param_dtype,
    #   )(sigma)
    
    emb = emb_time.astype(self.dtype)
    
    
    # Downsampling stack.
    skips = DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        num_input_proj_channels=self.input_proj_channels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_spatial_attention=use_spatial_attn,
        # use_temporal_attention=use_temporal_attn,
        use_position_encoding=self.use_position_encoding,
        ffn_type=self.ffn_type,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        num_heads=self.num_heads,
        normalize_qk=self.normalize_qk,
    )(x, emb, is_training)
    # Upsampling stack.
    h = UStack(
        num_channels=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        upsample_ratio=self.downsample_ratio[::-1],
        num_output_proj_channels=self.output_proj_channels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_spatial_attention=use_spatial_attn[::-1],
        # use_temporal_attention=use_temporal_attn[::-1],
        num_heads=self.num_heads,
        ffn_type=self.ffn_type,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        normalize_qk=self.normalize_qk,
    )(skips[-1], emb, skips, is_training)

    # Final convolution.
    h = nn.swish(
        nn.GroupNorm(
            min(h.shape[-1] // 4, 32),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(h)
    )
    h = ConvLayer(
        features=self.out_channels,
        kernel_size=tuple(self.kernel_size),
        padding=self.padding,
        kernel_init=default_init(),  # The kernel has mean almost zero.
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="conv3d_out",
    )(h)

    # If input was resized, resize the output to the original input shape.
    # if self.resize_to_shape is not None:
    #   h = FilteredResize(
    #       output_size=input_size,
    #       kernel_size=(7, 7),
    #       padding=self.padding,
    #       precision=self.precision,
    #       dtype=self.dtype,
    #       param_dtype=self.param_dtype,
    #   )(h)
    return h.astype(orig_dtype)
