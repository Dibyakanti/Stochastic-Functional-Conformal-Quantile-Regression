from flax import linen as nn


Initializer = nn.initializers.Initializer

def default_init(scale: float = 1e-10) -> Initializer:
  return nn.initializers.variance_scaling(
      scale=scale, mode="fan_avg", distribution="uniform"
  )

def as_3d_ratio(ratio: int | tuple[int, int, int]) -> tuple[int, int, int]:
  """Broadcasts an isotropic int ratio to (rx, ry, rz); tuples pass through.

  Per-axis ratios such as (2, 2, 1) let the 3D stacks run on effectively 2D
  data stored with a singleton z axis, i.e. (batch, x, y, 1, channels).
  """
  if isinstance(ratio, int):
    return (ratio, ratio, ratio)
  ratio = tuple(int(r) for r in ratio)
  if len(ratio) != 3:
    raise ValueError(f"Per-axis ratio must have 3 entries, got {ratio}.")
  return ratio

