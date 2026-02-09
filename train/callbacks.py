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

"""Training callback library."""

from collections.abc import Mapping, Sequence
import functools

from clu import metric_writers
import jax
import numpy as np
import orbax.checkpoint as ocp
import os
import tqdm

from train import trainers
from train import utils


Array = jax.Array
ComputedMetrics = Mapping[str, Array | Mapping[str, Array]]
Trainer = trainers.BaseTrainer


def to_shard_shape_dtype(
    x: Array, replicated: bool = False
) -> jax.ShapeDtypeStruct:
  aval = jax.api_util.shaped_abstractify(x)
  if replicated:
    return jax.ShapeDtypeStruct(aval.shape[1:], dtype=aval.dtype)
  else:
    return jax.ShapeDtypeStruct(aval.shape, dtype=aval.dtype)

class Callback:
  """Abstract base class for callbacks.

  Callbacks are self-contained programs containing some common, reusable logic
  that is non-essential (such as saving model checkpoints, reporting progress,
  profiling, the absence of which would not "break" training) to model training.
  The instance methods of these objects are hooks that get executed at various
  phases of training (i.e. fixed positions inside `train.run` function).

  The execution (in `train.run`) observes the following flow::

    callbacks.on_train_begin()
    while training:
      callbacks.on_train_batches_begin()
      run_train_steps()
      callbacks.on_train_batches_end()
      if should_run_evaluation:
        callbacks.on_eval_batches_begin()
        run_eval_steps()
        callbacks.on_eval_batches_end()
    callbacks.on_train_end()

  The hooks may read and/or overwrite the trainer state and/or train/eval
  metrics, and have access to a metric_writer that writes desired info/variables
  to the working directory in tensorflow event format.

  When multiple (i.e. a list of) callbacks are used, the
  `on_{train/eval}_batches_end` methods are called in reverse order (so that
  together with `on_{train/eval}_batches_begin` calls they resemble
  the `__exit__` and `__enter__` methods of python contexts).
  """

  @property
  def metric_writer(self) -> metric_writers.MultiWriter:
    assert hasattr(self, "_metric_writer")
    return self._metric_writer

  @metric_writer.setter
  def metric_writer(self, writer: metric_writers.MultiWriter) -> None:
    self._metric_writer = writer

  def on_train_begin(self, trainer: Trainer) -> None:
    """Called before the training loop starts."""

  def on_train_batches_begin(self, trainer: Trainer) -> None:
    """Called before a training segment begins."""

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: ComputedMetrics
  ) -> None:
    """Called after a training segment ends."""

  def on_eval_batches_begin(self, trainer: Trainer) -> None:
    """Called before an evaluation segment begins."""

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: ComputedMetrics
  ) -> None:
    """Called after an evaluation segment ends."""

  def on_train_end(self, trainer: Trainer) -> None:
    """Called when training ends."""
    
# This callback does not seem to work with `utils.primary_process_only`.
class TrainStateCheckpoint(Callback):
  """Callback that periodically saves train state checkpoints."""

  def __init__(
      self,
      base_dir: str,
      folder_prefix: str = "checkpoints",
      train_state_field: str = "default",
      options: ocp.CheckpointManagerOptions | None = None,
  ):
    self.save_dir = os.path.join(base_dir, folder_prefix)
    self.train_state_field = train_state_field
    self.ckpt_manager = ocp.CheckpointManager(
        self.save_dir,
        item_handlers={self.train_state_field: ocp.StandardCheckpointHandler()},
        options=options,
    )

  def on_train_begin(self, trainer: Trainer) -> None:
    """Sets up directory, saves initial or restore the most recent state."""
    self.last_eval_metric = {}
    # retrieve from existing checkpoints if possible
    if self.ckpt_manager.latest_step() is not None:
      # Load a single shard and then replicate explicitly.
      map_fn = functools.partial(
          to_shard_shape_dtype, replicated=trainer.is_distributed
      )
      restored = self.ckpt_manager.restore(
          self.ckpt_manager.latest_step(),
          args=ocp.args.Composite(**{
              self.train_state_field: ocp.args.StandardRestore(
                  item=jax.tree.map(map_fn, trainer.train_state)
              )
          }),
      )

      trainer.train_state = trainer._maybe_replicate(  # pylint: disable=protected-access
          getattr(restored, self.train_state_field)
      )

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: ComputedMetrics
  ) -> None:
    assert self.last_eval_metric is not None
    cur_step = trainer.train_state.int_step
    scalar_train_metrics = {
        k: float(v) for k, v in train_metrics.items() if utils.is_scalar(v)
    }
    if self.ckpt_manager.should_save(cur_step):
      self.ckpt_manager.save(
          step=cur_step,
          # This always saves the unreplicated train state.
          # Converting to np array seems necessary for multi-host environments.
          args=ocp.args.Composite(**{
              self.train_state_field: ocp.args.StandardSave(
                  jax.tree.map(np.array, trainer.unreplicated_train_state)
              )
          }),
          metrics=dict(**scalar_train_metrics, **self.last_eval_metric),
      )

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: ComputedMetrics
  ) -> None:
    del trainer
    self.last_eval_metric = {
        k: float(v) for k, v in eval_metrics.items() if utils.is_scalar(v)
    }

  def on_train_end(self, trainer: Trainer) -> None:
    # Always save a checkpoint at the end of training.
    if self.ckpt_manager.latest_step() != trainer.train_state.int_step:
      self.ckpt_manager.save(
          trainer.train_state.int_step,
          args=ocp.args.Composite(**{
              self.train_state_field: ocp.args.StandardSave(
                  jax.tree.map(np.array, trainer.unreplicated_train_state)
              )
          }),
          force=True,
      )
    self.ckpt_manager.wait_until_finished()

class TqdmProgressBar(Callback):
  """Tqdm progress bar callback to monitor training progress in real time."""

  def __init__(
      self,
      total_train_steps: int | None,
      train_monitors: Sequence[str],
      eval_monitors: Sequence[str] = (),
  ):
    """ProgressBar constructor.

    Args:
      total_train_steps: the total number of training steps, which is displayed
        as the maximum progress on the bar.
      train_monitors: keys in the training metrics whose values are updated on
        the progress bar after every training metric aggregation.
      eval_monitors: same as `train_monitors` except applying to evaluation.
    """
    self.total_train_steps = total_train_steps
    self.train_monitors = train_monitors
    self.eval_monitors = eval_monitors
    self.current_step = 0
    self.eval_postfix = {}  # keeps record of the most recent eval monitor
    self.bar = None

  def on_train_begin(self, trainer: Trainer) -> None:
    del trainer
    self.bar = tqdm.tqdm(total=self.total_train_steps, unit="step")

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: ComputedMetrics
  ) -> None:
    assert self.bar is not None
    self.bar.update(trainer.train_state.int_step - self.current_step)
    self.current_step = trainer.train_state.int_step
    postfix = {
        monitor: train_metrics[monitor] for monitor in self.train_monitors
    }
    self.bar.set_postfix(**postfix, **self.eval_postfix)

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: ComputedMetrics
  ) -> None:
    del trainer
    self.eval_postfix = {
        monitor: eval_metrics[monitor] for monitor in self.eval_monitors
    }

  def on_train_end(self, trainer: Trainer) -> None:
    del trainer
    assert self.bar is not None
    self.bar.close()