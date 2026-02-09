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


Array = jax.Array
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)

class FourierEmbedding(nn.Module):
  """Fourier embedding.

  Attributes:
    dims: The number of dimensions in the embedding.
    max_freq: The maximum frequency of the embedding.
    projection: Whether to project the embedding to the desired output
      dimension.
    act_fun: The activation function.
    precision: The precision of the computation.
    dtype: The data type of the input/output.
    param_dtype: The data type of the parameters.
  """

  dims: int = 64
  max_freq: float = 2e4
  projection: bool = True
  use_magnitude_preserving: bool = False
  act_fun: Callable[[Array], Array] = nn.swish
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    if x.ndim != 1:
      raise ValueError(
          f"Input tensor must be 1D. Instead it has {x.ndim} dimensions."
      )
    logfreqs = jnp.linspace(
        0.0, jnp.log(self.max_freq), self.dims // 2, dtype=jnp.float32
    )
    # Cast to float32 to avoid numerical issues for the sin/cos operations.
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    x = jnp.pi * jnp.exp(logfreqs)[None, :] * x[:, None]
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    if self.use_magnitude_preserving:
      # We multiply by sqrt(2) to ensure that the norm of the embedding is
      # preserved.
      x *= jnp.sqrt(2.0)

    # Project to the desired output dimension.
    if self.projection:
      x = nn.Dense(
          features=2 * self.dims,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x)
      x = self.act_fun(x)
      x = nn.Dense(
          features=self.dims,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x)

    # Cast back to the original dtype.
    x = x.astype(orig_dtype)

    return x
