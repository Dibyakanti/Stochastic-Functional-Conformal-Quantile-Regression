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

from collections.abc import Callable, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from layers.convolutions import ConvLayer

Array = jax.Array
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)

class FilteredResize(nn.Module):
  """A resizing op followed by a convolution layer.

  Attributes:
    output_size: The target output spatial dimensions for resizing.
    kernel_size: The kernel size of the convolution layer.
    method: The resizing method (passed to `jax.image.resize`).
    padding: The padding type of the convolutions, one of ['SAME', 'CIRCULAR',
      'LATLON', 'LONLAT].
    initializer: The initializer for the convolution kernels.
    use_local: Whether to use unshared weights in the filtering.
    precision: Level of precision used in the convolutional layer.
    dtype: The data type of the input and output.
    params_dtype: The data type of of the weights.
  """

  output_size: Sequence[int]
  kernel_size: Sequence[int]
  method: str = "cubic"
  padding: str = "CIRCULAR"
  initializer: nn.initializers.Initializer = jax.nn.initializers.normal(
      stddev=0.02
  )
  use_local: bool = False
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Resizes and filters the input with a convolution layer.

    Args:
      inputs: An input tensor of shape `(*batch_dims, *resized_dims, channels)`,
        where `batch_dims` can be or arbitrary length and `resized_dims` has the
        same length as that of `self.kernel_size`.

    Returns:
      The input resized to target shape.
    """
    if not inputs.ndim > len(self.output_size):
      raise ValueError(
          f"Number of dimensions in x ({inputs.ndim}) must be larger than the"
          f" length of `output_size` ({len(self.output_size)})!"
      )

    if self.padding not in ["CIRCULAR", "LATLON", "LONLAT", "SAME"]:
      raise ValueError(
          f"Unsupported padding type: {self.padding} - please use one of"
          " ['SAME', 'CIRCULAR', 'LATLON', 'LONLAT']!"
      )

    batch_ndim = inputs.ndim - len(self.output_size) - 1
    orig_dtype = inputs.dtype
    # We perform the resizing step at input accuracy.
    # TODO: Perhaps it makes more sense to use the resize at
    # single precision and then cast to the desired type.
    resized = jax.image.resize(
        inputs,
        shape=(*inputs.shape[:batch_ndim], *self.output_size, inputs.shape[-1]),
        method=self.method,
    )

    resized = resized.astype(self.dtype)
    # We add another convolution layer to undo any aliasing that could have
    # been introduced by the resizing step.
    out = ConvLayer(
        features=inputs.shape[-1],
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
        dtype=self.dtype,
        precision=self.precision,
        param_dtype=self.param_dtype,
    )(resized)
    return out.astype(orig_dtype)

class MergeChannelCond(nn.Module):
  """Base class for merging conditional inputs along the channel dimension.

  Attributes:
    embed_dim: The output channel dimension.
    kernel_size: The convolutional kernel size.
    resize_method: The interpolation method employed by `jax.image.resize`.
    padding: The padding method of all convolutions.
  """

  embed_dim: int
  kernel_size: Sequence[int]
  resize_method: str = "cubic"
  padding: str = "CIRCULAR"
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32


class InterpConvMerge(MergeChannelCond):
  """Merges conditional inputs through interpolation and convolutions.

  See `MergeChannelCond` for attribute definition.
  """

  @nn.compact
  def __call__(self, x: Array, cond: dict[str, Array]):
    """Merges conditional inputs along the channel dimension.

    Fields with spatial shape differing from the sample `x` are reshaped to
    match it. Then, all conditional fields are passed through a nonlinearity,
    a ConvLayer, and then concatenated with the main input along the last axis.

    Args:
      x: The main model input.
      cond: A dictionary of conditional inputs. Those with keys that start with
        "channel:" are processed here while all others are omitted.

    Returns:
      Model input merged with channel conditions.
    """

    out_spatial_shape = x.shape[-3:-1]
    for key, value in sorted(cond.items()):
      if key.startswith("channel:"):
        if value.ndim != x.ndim:
          raise ValueError(
              f"Channel condition `{key}` does not have the same ndim"
              f" ({value.ndim}) as x ({x.ndim})!"
          )

      if value.shape[-3:-1] != out_spatial_shape:
        value = FilteredResize(
            output_size=x.shape[2:],
            kernel_size=self.kernel_size,
            method=self.resize_method,
            padding=self.padding,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"resize_{key}",
        )(value)
      value = nn.swish(nn.LayerNorm()(value))
      value = ConvLayer(
          features=self.embed_dim,
          kernel_size=self.kernel_size,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"conv2d_embed_{key}",
      )(value)

      # Cast to the dtype of the input to avoid a type mismatch when
      # concatenating.
      value = value.astype(x.dtype)
      x = jnp.concatenate([x, value], axis=-1, dtype=x.dtype)

    return x