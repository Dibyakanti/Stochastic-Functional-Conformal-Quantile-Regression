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

from collections.abc import Callable
from typing import Any
import functools

from clu import values
import jax
import jax.numpy as jnp
import numpy as np

Scalar = Any


def primary_process_only(cls: type[Any]) -> type[Any]:
  """Class decorator that modifies all methods to run on primary host only."""

  def wrap_method(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
      if jax.process_index() == 0:
        return method(self, *args, **kwargs)
      else:
        return None

    return wrapper

  for attr_name, attr_value in cls.__dict__.items():
    if callable(attr_value) and not attr_name.startswith("__"):
      setattr(cls, attr_name, wrap_method(attr_value))

  return cls

def is_scalar(value: Any) -> bool:
  """Checks if a given value is a scalar."""
  if isinstance(value, values.Scalar) or isinstance(
      value, (int, float, np.number)
  ):
    return True
  if isinstance(value, (np.ndarray, jnp.ndarray)):
    return value.ndim == 0 or value.size <= 1
  return False