from flax import linen as nn


Initializer = nn.initializers.Initializer

def default_init(scale: float = 1e-10) -> Initializer:
  return nn.initializers.variance_scaling(
      scale=scale, mode="fan_avg", distribution="uniform"
  )