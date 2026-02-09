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

from layers.residual import CombineResidualWithSkip
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

class AddAxialPositionEmbedding(nn.Module):
  """Adds trainable axial position embeddings to the inputs."""

  position_axis: int
  initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    pos_axis = self.position_axis
    pos_axis = pos_axis if pos_axis >= 0 else pos_axis + inputs.ndim

    if not 0 <= pos_axis < inputs.ndim:
      raise ValueError(
          f"Invalid position ({self.position_axis}) or feature axis"
          f" ({self.feature_axis})!"
      )

    feat_axis = inputs.ndim - 1
    if pos_axis == feat_axis:
      raise ValueError(
          f"Position axis ({self.position_axis}) must not coincide with feature"
          f" axis ({feat_axis})!"
      )

    unsqueeze_axes = tuple(set(range(inputs.ndim)) - {pos_axis, feat_axis})
    embedding = self.param(
        "embedding",
        self.initializer,
        (inputs.shape[pos_axis], inputs.shape[feat_axis]),
    )
    return inputs + jnp.expand_dims(embedding, axis=unsqueeze_axes)


class AxialSelfAttention(nn.Module):
  """Axial self-attention for multidimensional inputs."""

  num_heads: int
  attention_axis: int = -2
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
  deterministic: bool = True
  precision: PrecisionLike = None
  normalize_qk: bool = False
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies axial self-attention to the inputs."""

    if self.attention_axis == -1 or self.attention_axis == inputs.ndim - 1:
      raise ValueError(
          f"Attention axis ({self.attention_axis}) cannot be the last axis,"
          " which is treated as the features!"
      )
    # Casting to the specified dtype and saving the original dtype.
    orig_dtype = inputs.dtype
    inputs = inputs.astype(self.dtype)

    inputs = jnp.swapaxes(inputs, self.attention_axis, -2)
    inputs_q = jnp.reshape(inputs, (-1, *inputs.shape[-2:]))

    out = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        deterministic=self.deterministic,
        normalize_qk=self.normalize_qk,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
    )(inputs_q=inputs_q)

    out = jnp.reshape(out, inputs.shape)
    out = jnp.swapaxes(out, -2, self.attention_axis)

    return out.astype(orig_dtype)

class AxialSelfAttentionBlock(nn.Module):
  """Block consisting of (potentially multiple) axial attention layers."""

  attention_axes: int | Sequence[int] = -2
  add_position_embedding: bool | Sequence[bool] = True
  num_heads: int | Sequence[int] = 1
  precision: PrecisionLike = None
  normalize_qk: bool = False
  ffn_type: Literal["dense", "swiglu", "geglu", "rational_glu"] = "dense"
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, is_training: bool) -> Array:

    # Casting to the specified dtype.
    orig_dtype = x.dtype
    x = x.astype(self.dtype)

    # Consolidate configs for all attention layers.
    attn_axes = self.attention_axes
    attn_axes = (attn_axes,) if isinstance(attn_axes, int) else attn_axes
    num_axes = len(attn_axes)

    pos_embed = self.add_position_embedding
    if isinstance(pos_embed, bool):
      pos_embed = (pos_embed,) * num_axes

    num_heads = self.num_heads
    if isinstance(num_heads, int):
      num_heads = (num_heads,) * num_axes

    # Axial attention ops followed by a projection.
    h = x
    for axis, add_emb, num_head in zip(attn_axes, pos_embed, num_heads):
      if add_emb:
        h = AddAxialPositionEmbedding(
            position_axis=axis, name=f"pos_emb_axis{axis}"
        )(h)
      # Normalization layers are processed in float32 to avoid numerical issues.
      h = nn.GroupNorm(
          min(h.shape[-1] // 4, 32),
          name=f"attn_prenorm_axis{axis}",
          dtype=jnp.float32,
          param_dtype=jnp.float32,
      )(h)
      h = AxialSelfAttention(
          attention_axis=axis,
          num_heads=num_head,
          kernel_init=nn.initializers.xavier_uniform(),
          deterministic=not is_training,
          normalize_qk=self.normalize_qk,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
          name=f"axial_attn_axis{axis}",
      )(h)
      h = nn.GroupNorm(
          min(h.shape[-1] // 4, 32),
          name=f"dense_prenorm_axis{axis}",
          dtype=jnp.float32,  # Normalization is done in float32.
          param_dtype=jnp.float32,
      )(h)

      # The different types of Feeding Forward Networks (FFNs).
      # if self.ffn_type == "dense":
      h = nn.Dense(
          features=h.shape[-1],
          kernel_init=default_init(1.0),
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(h)
      # else:
      #   match self.ffn_type:
      #     case "swiglu":
      #       act_fun = nn.swish
      #     case "geglu":
      #       act_fun = nn.gelu
      #     case "rational_glu":
      #       act_fun = rational_networks.RationalLayer(dtype=self.dtype)
      #     case _:
      #       raise ValueError(f"Unsupported ffn_type: {self.ffn_type}")

      #   h = unets.ResConv1xGLU(
      #       hidden_layer_size=3 * int(h.shape[-1] // 2),
      #       out_channels=h.shape[-1],
      #       act_fun=act_fun,
      #       dtype=self.dtype,
      #       param_dtype=self.param_dtype,
      #       precision=self.precision,
      #   )(h)

    y = CombineResidualWithSkip()(residual=h, skip=x)
    return y.astype(orig_dtype)