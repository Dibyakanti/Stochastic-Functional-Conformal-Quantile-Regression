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

"""Generic model class for use in gradient descent mini-batch training."""
import dataclasses
import abc
from collections.abc import Callable, Mapping
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

ArrayDict = Mapping[str, jax.Array]
BatchType = Mapping[str, np.ndarray | jax.Array]
VarDict = nn.module.FrozenVariableDict
ModelVariable = VarDict | tuple[VarDict, ...] | Mapping[str, VarDict]
PyTree = Any
LossAndAux = tuple[jax.Array, tuple[ArrayDict, PyTree]]


class BaseModel(metaclass=abc.ABCMeta):
  """Base class for models.

  Wraps flax module(s) to provide interfaces for variable
  initialization, computing loss and evaluation metrics. These interfaces are
  to be used by a trainer to perform gradient updates as it steps through the
  batches of a dataset.

  Subclasses must implement the abstract methods.
  """

  @abc.abstractmethod
  def initialize(self, rng: jax.Array) -> ModelVariable:
    """Initializes variables of the wrapped flax module(s).

    This method by design does not take any sample input in its argument. Input
    shapes are expected to be statically known and used to create
    initialization input for the model. For example::

      import flax.linen as nn
      import jax.numpy as jnp

      class MLP(BaseModel):
        def __init__(self, mlp: nn.Module, input_shape: tuple[int]):
          self.model = mlp
          self.input_shape = input_shape

        def initialize(self, rng):
          init_input = jnp.ones(self.input_shape)
          return self.model.init(rng, init_input)

    Args:
      rng: random key used for initialization.

    Returns:
      The initial variables for this model - can be a single or a tuple/mapping
      of flax variables.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def loss_fn(
      self,
      params: PyTree | tuple[PyTree, ...],
      batch: BatchType,
      rng: jax.Array,
      mutables: PyTree,
      **kwargs,
  ) -> LossAndAux:
    """Computes training loss and metrics.

    It is expected that gradient would be taken (via `jax.grad`) wrt `params`
    during training.

    Arguments:
      params: model parameters wrt which the loss would be differentiated.
      batch: a single batch of data.
      rng: jax random key for randomized loss if needed.
      mutables: model variables which are not differentiated against; can be
        mutable if so desired.
      **kwargs: additional static configs.

    Returns:
      loss: the (scalar) loss function value.
      aux: two-item auxiliary data consisting of
        metric_vars: a dict with values required for metric compute and logging.
          They can either be final metric values computed inside the function or
          intermediate values to be further processed into metrics.
        mutables: non-differentiated model variables whose values may change
          during function execution (e.g. batch stats).
    """
    raise NotImplementedError

  def eval_fn(
      self,
      variables: tuple[PyTree, ...] | PyTree,
      batch: BatchType,
      rng: jax.Array,
      **kwargs,
  ) -> ArrayDict:
    """Computes evaluation metrics."""
    raise NotImplementedError

  @staticmethod
  def inference_fn(variables: PyTree, **kwargs) -> Callable[..., Any]:
    """Returns an inference function with bound variables."""
    raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataAssimilationModel(BaseModel):
  """
  Trains a model to predict some field at the next time-step.
  
  Additional Attributes:
    model : The model to be used for the data assimilation task (eg : UNet3D)
  """
  model: Any
  input_shape: tuple[int, ...]

  def initialize(self, rng: jax.Array) -> Any:
    """Pure initialization without internal state."""
    dummy_x = jnp.zeros(self.input_shape)
    # Match the batch dimension for time and sigma
    batch_size = self.input_shape[0]
    return self.model.init(
        rng, 
        dummy_x, 
        time=jnp.zeros((batch_size,)), 
        is_training=False
    )

  def loss_fn(
      self,
      params: Any,
      batch: dict[str, jax.Array],
      rng: jax.Array,
      mutables: Any,
  ) -> LossAndAux:
    """L2 loss function."""
    variables = {"params": params, **mutables}
    
    # We pass the dropout RNG if provided
    preds = self.model.apply(
        variables,
        batch["x"],
        time=batch["time"],
        is_training=True,
        rngs={"dropout": rng} if rng is not None else None
    )

    loss = jnp.mean(jnp.square(preds - batch["y"]))
    return loss, ({"loss": loss}, mutables)

  def eval_fn(
      self,
      variables: Any,
      batch: dict[str, jax.Array],
      rng: jax.Array,
  ) -> dict[str, jax.Array]:
    """Evaluation function."""
    preds = self.model.apply(
        variables,
        batch["x"],
        time=batch["time"],
        is_training=False
    )
    mse = jnp.mean(jnp.square(preds - batch["y"]))
    return {"mse": mse}
  
@dataclasses.dataclass(frozen=True, kw_only=True)
class QuantileEnsembleDAModel(BaseModel):
  model: Any
  input_shape: tuple[int, ...]
  n_ens: int = 1
  tau: float = 0.5

  def initialize(self, rng: jax.Array) -> Any:
    # Initialize 'n_ens' independent sets of parameters
    dummy_x = jnp.zeros(self.input_shape)
    batch_size = self.input_shape[0]
    init_keys = jax.random.split(rng, self.n_ens)
    
    return jax.vmap(lambda k: self.model.init(
        k, dummy_x, time=jnp.zeros((batch_size,)), is_training=False
    ))(init_keys)

  def loss_fn(self, params: Any, batch: dict[str, jax.Array], rng: jax.Array, mutables: Any) -> LossAndAux:
    def single_loss(p, m):
      preds = self.model.apply({"params": p, **m}, batch["x"], 
                               time=batch["time"], is_training=True,
                               rngs={"dropout": rng} if rng is not None else None)
      err = batch["y"] - preds
      # Pinball loss: max(tau*err, (tau-1)*err)
      return jnp.mean(jnp.maximum(self.tau * err, (self.tau - 1) * err))

    # Average loss across the ensemble
    losses = jax.vmap(single_loss)(params, mutables)
    loss = jnp.mean(losses)
    return loss, ({"loss": loss}, mutables)

  def eval_fn(self, variables: Any, batch: dict[str, jax.Array], rng: jax.Array) -> dict[str, jax.Array]:
    # vmap apply to get all ensemble predictions
    preds = jax.vmap(lambda p, m: self.model.apply(
        {"params": p, **m}, batch["x"], time=batch["time"], is_training=False
    ))(variables["params"], variables.get("mutables", {}))
    
    # Checkpoint metrics: use the mean of the ensemble for MSE
    avg_pred = jnp.mean(preds, axis=0)
    mse = jnp.mean(jnp.square(avg_pred - batch["y"]))
    return {"mse": mse}