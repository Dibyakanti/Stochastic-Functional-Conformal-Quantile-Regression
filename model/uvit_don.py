"""DeepONet whose branch net is the UViT encoder of `UNet3d`.

The branch takes the whole input field (batch, x, y, z, channels) and the time
gap, and runs exactly the encoder half of `model/unet3d.py`: a Fourier
embedding of the time gap (fed to every conv block through FiLM) and the
`DStack` downsampling stack (conv blocks, with axial self-attention at the
levels where `use_spatial_attention` is set). The bottleneck feature map
(`skips[-1]`) is flattened and mapped by one Dense layer to the n_out * p
branch latents. The trunk is the usual MLP on (x, y, z). Output c is the dot
product of the c-th group of p branch and trunk latents ("split both"), plus
an optional learned per-output bias (the b_0 of Lu et al.'s DeepONet).
"""

from collections.abc import Sequence
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import optax
from tqdm import trange

from embeddings.fourier_emb import FourierEmbedding
from model.don import MLP, quantile_loss
from model.unet3d import _maybe_broadcast_to_list
from stacks.dstack3d import DStack


class UViTBranch(nn.Module):
  """UNet3d encoder (time embedding + DStack) followed by a linear head."""

  out_features: int
  num_channels: Sequence[int] = (16, 32, 64)
  downsample_ratio: Sequence[int] = (2, 2, 2)
  use_spatial_attention: bool | Sequence[bool] = (False, False, True)
  num_blocks: int = 4
  noise_embed_dim: int = 128
  input_proj_channels: int = 128
  padding: str = "SAME"
  use_position_encoding: bool = True
  num_heads: int = 8

  @nn.compact
  def __call__(self, x, time):
    # Same construction as UNet3d.__call__ up to the end of the DStack.
    emb = FourierEmbedding(dims=self.noise_embed_dim)(time)
    skips = DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        num_input_proj_channels=self.input_proj_channels,
        padding=self.padding,
        use_spatial_attention=_maybe_broadcast_to_list(
            self.use_spatial_attention, self.num_channels
        ),
        use_position_encoding=self.use_position_encoding,
        num_heads=self.num_heads,
    )(x, emb, is_training=False)  # dropout is 0, so training mode is unused
    h = skips[-1].reshape(x.shape[0], -1)
    return nn.Dense(self.out_features)(h)


class UViTDeepONetEnsemble:
  """Ensemble of UViT-branch DeepONets trained in parallel with vmap.

  Mirrors `DeepONetEnsemble` (params, opt_state, loss_don_log, train), but a
  batch is grouped by function so the branch runs once per field:
    fields (n_fn, X, Y, Z, C), time (n_fn,), y (n_fn, n_pts, 3),
    s (n_fn, n_pts, n_out).
  """

  def __init__(self, n_ens, branch_kwargs, field_shape, trunk_layers, lr,
               n_out, alpha=0.5, activation=jnp.tanh, rng_key=random.PRNGKey(0),
               output_bias=False, input_scale=1.0, total_steps=None,
               lr_schedule="constant", warmup_steps=0, weight_decay=0.0,
               clip_norm=0.0, mse_warmup_steps=0, decay_start=0.5,
               ema_decay=0.0):
    """The defaults give plain Adam at a constant lr with no output bias.

    lr_schedule: "constant"; "cosine" (linear warmup over `warmup_steps`,
      then cosine decay to 0 at `total_steps`); or "hold_cosine" (constant
      lr until `decay_start * total_steps`, then cosine decay to 0).
    ema_decay > 0 keeps an exponential moving average of the parameters,
      which replaces `params` at the end of `train` (and is what is saved).
    weight_decay > 0 uses AdamW; clip_norm > 0 clips the global grad norm.
    input_scale multiplies the branch input field.
    mse_warmup_steps: train on MSE for this many of the first steps (learning
      the conditional mean), then on the quantile loss. With additive noise
      the tau-quantile is the mean plus a constant, so the quantile phase
      mostly has to learn that offset.
    """
    self.n_ens = n_ens
    self.n_out = n_out
    self.alpha = alpha
    self.output_bias = output_bias
    self.input_scale = input_scale
    self.mse_warmup_steps = mse_warmup_steps
    self.ema_decay = ema_decay
    self.step = 0

    # trunk_layers[-1] is the latent size p per output
    p = trunk_layers[-1]
    self.branch = UViTBranch(out_features=n_out * p, **branch_kwargs)
    trunk_layers = list(trunk_layers[:-1]) + [n_out * p]
    self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=activation)

    dummy_x = jnp.zeros((1, *field_shape))
    dummy_t = jnp.zeros((1,))

    def _init_one(key):
      k1, k2 = random.split(key)
      # MLP init follows jax_enable_x64; keep the trunk in float32 like the
      # branch (float64 is very slow on most GPUs).
      trunk = jax.tree.map(lambda a: a.astype(jnp.float32),
                           self.trunk_init(rng_key=k2))
      params = (self.branch.init(k1, dummy_x, dummy_t)["params"], trunk)
      if output_bias:
        params = params + (jnp.zeros((n_out,), jnp.float32),)
      return params
    self.params = vmap(_init_one)(random.split(rng_key, n_ens))

    if lr_schedule == "constant":
      schedule = lr
    elif lr_schedule == "cosine":
      schedule = optax.warmup_cosine_decay_schedule(
          init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
          decay_steps=total_steps, end_value=0.0,
      )
    elif lr_schedule == "hold_cosine":
      hold = int(decay_start * total_steps)
      schedule = optax.join_schedules(
          [optax.constant_schedule(lr),
           optax.cosine_decay_schedule(lr, total_steps - hold)],
          [hold],
      )
    else:
      raise ValueError(f"Unknown lr_schedule: {lr_schedule}")
    opt = (optax.adamw(schedule, weight_decay=weight_decay) if weight_decay > 0
           else optax.adam(schedule))
    self.optimizer = (optax.chain(optax.clip_by_global_norm(clip_norm), opt)
                      if clip_norm > 0 else opt)
    self.opt_state = vmap(self.optimizer.init)(self.params)
    self.ema_params = self.params
    self.raw_params = self.params

    self.loss_don_log = []

  # ---- Core network ----
  def branch_latents(self, params, fields, time):
    B = self.branch.apply({"params": params[0]}, self.input_scale * fields, time)
    return B.reshape(fields.shape[0], self.n_out, -1)  # (n_fn, n_out, p)

  def trunk_latents(self, params, y):
    T = self.trunk_apply(params[1], y)
    return T.reshape(*y.shape[:-1], self.n_out, -1)    # (..., n_out, p)

  def predict(self, params, fields, time, y):
    """Per-function points y (n_fn, n_pts, 3) -> (n_fn, n_pts, n_out)."""
    B = self.branch_latents(params, fields, time)
    T = self.trunk_latents(params, y)
    return self._add_bias(params, jnp.einsum("fcp,fncp->fnc", B, T))

  def predict_grid(self, params, fields, time, grid):
    """One shared grid (n_grid, 3) -> (n_fn, n_grid, n_out)."""
    B = self.branch_latents(params, fields, time)
    T = self.trunk_latents(params, grid)
    return self._add_bias(params, jnp.einsum("fcp,gcp->fgc", B, T))

  def _add_bias(self, params, out):
    return out + params[2] if self.output_bias else out

  # ---- Loss ----
  def loss_don(self, params, batch, mse=False):
    fields, time, y, s = batch
    pred = self.predict(params, fields, time, y)
    if mse:
      return jnp.mean((s - pred) ** 2)
    return jnp.mean(quantile_loss(self.alpha, s, pred))

  @partial(jit, static_argnums=(0, 4))
  def _step_all(self, params, opt_state, batch, mse=False):
    def step_one(params, opt_state, batch):
      loss, g = jax.value_and_grad(self.loss_don)(params, batch, mse)
      updates, opt_state = self.optimizer.update(g, opt_state, params)
      return optax.apply_updates(params, updates), opt_state, loss
    return vmap(step_one)(params, opt_state, batch)

  # ---- Training loop ----
  def train(self, sampler, nIter, log_every=20):
    """`sampler.next_batch()` returns a batch with a leading n_ens axis."""
    if self.ema_decay > 0:
      self.params = self.raw_params  # resume from the raw, not EMA, weights
    pbar = trange(nIter, desc="Training Ensemble")
    for it in pbar:
      batch = sampler.next_batch()
      self.params, self.opt_state, losses = self._step_all(
          self.params, self.opt_state, batch, self.step < self.mse_warmup_steps
      )
      if self.ema_decay > 0:
        self.ema_params = self._ema(self.ema_params, self.params)
      self.step += 1
      if it % log_every == 0:
        # Mean over members of the loss on the batch before this step.
        train_loss = float(jnp.mean(losses))
        self.loss_don_log.append(train_loss)
        pbar.set_postfix({"train_loss": train_loss})
    self.raw_params = self.params
    if self.ema_decay > 0:
      self.params = self.ema_params

  @partial(jit, static_argnums=(0,))
  def _ema(self, ema, params):
    d = self.ema_decay
    return jax.tree.map(lambda e, p: d * e + (1 - d) * p, ema, params)


class FunctionPointSampler:
  """Independent minibatches per ensemble member: `n_fn` distinct functions,
  each with `n_pts` points drawn from that function's own points.

  fields (N, X, Y, Z, C), time (N,), y (N, P, 3), s (N, P, n_out).
  """

  def __init__(self, fields, time, y, s, n_ens, n_fn, n_pts, rng_key):
    self.data = (jnp.asarray(fields), jnp.asarray(time), jnp.asarray(y),
                 jnp.asarray(s))
    self.key = rng_key
    N, P = y.shape[0], y.shape[1]
    n_fn = min(n_fn, N)

    def sample_one(key, data):
      fields, time, y, s = data
      k1, k2 = random.split(key)
      fi = random.choice(k1, N, (n_fn,), replace=False)
      pi = random.randint(k2, (n_fn, n_pts), 0, P)
      return (fields[fi], time[fi], y[fi[:, None], pi], s[fi[:, None], pi])

    self._sample = jit(lambda keys, data: vmap(sample_one, (0, None))(keys, data))
    self.n_ens = n_ens

  def next_batch(self):
    self.key, sub = random.split(self.key)
    return self._sample(random.split(sub, self.n_ens), self.data)


def _cyclic(field, k):
  """Applies k times (x, y, z) -> (y, z, x) with (u, v, w) -> (v, w, u) to one
  field (X, Y, Z, 4); p is unchanged. The Beltrami solution with a = d is
  invariant under this map, and i.i.d. noise keeps its distribution."""
  def once(f):
    return jnp.transpose(f, (1, 2, 0, 3))[..., jnp.array([1, 2, 0, 3])]
  f1 = once(field)
  return jnp.stack([field, f1, once(f1)])[k]


class SymmetryAugmentedSampler:
  """Like `FunctionPointSampler`, but on whole fields: every sampled input and
  target field independently gets a random cyclic transform (`_cyclic`),
  i.e. a fresh, equally valid noise realisation of the same flow, and the
  points are drawn from the whole grid.

  fields, targets (N, X, Y, Z, 4), time (N,), grid (X*Y*Z, 3) in the same
  flat order as field.reshape(-1, 4).
  """

  def __init__(self, fields, time, targets, grid, n_ens, n_fn, n_pts, rng_key):
    self.data = (jnp.asarray(fields), jnp.asarray(time), jnp.asarray(targets),
                 jnp.asarray(grid))
    self.key = rng_key
    self.n_ens = n_ens
    N, G = fields.shape[0], grid.shape[0]
    n_fn = min(n_fn, N)

    def sample_one(key, data):
      fields, time, targets, grid = data
      k1, k2, k3, k4 = random.split(key, 4)
      fi = random.choice(k1, N, (n_fn,), replace=False)
      k_in = random.randint(k2, (n_fn,), 0, 3)
      k_out = random.randint(k3, (n_fn,), 0, 3)
      f = vmap(_cyclic)(fields[fi], k_in)
      tgt = vmap(_cyclic)(targets[fi], k_out).reshape(n_fn, G, -1)
      pi = random.randint(k4, (n_fn, n_pts), 0, G)
      return (f, time[fi], grid[pi], jnp.take_along_axis(tgt, pi[..., None], axis=1))

    self._sample = jit(lambda keys, data: vmap(sample_one, (0, None))(keys, data))

  def next_batch(self):
    self.key, sub = random.split(self.key)
    return self._sample(random.split(sub, self.n_ens), self.data)
