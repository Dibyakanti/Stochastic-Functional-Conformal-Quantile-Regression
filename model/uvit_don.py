"""DeepONet whose branch net is the UViT encoder of `UNet3d`.

The branch takes the whole input field (batch, x, y, z, channels) and the time
gap, and runs exactly the encoder half of `model/unet3d.py`: a Fourier
embedding of the time gap (fed to every conv block through FiLM) and the
`DStack` downsampling stack (conv blocks, with axial self-attention at the
levels where `use_spatial_attention` is set). The bottleneck feature map
(`skips[-1]`) is flattened and mapped by one Dense layer to the n_out * p
branch latents. The trunk is the usual MLP on (x, y, z). Output c is the dot
product of the c-th group of p branch and trunk latents ("split both").
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
               n_out, alpha=0.5, activation=jnp.tanh, rng_key=random.PRNGKey(0)):
    self.n_ens = n_ens
    self.n_out = n_out
    self.alpha = alpha

    # trunk_layers[-1] is the latent size p per output
    p = trunk_layers[-1]
    self.branch = UViTBranch(out_features=n_out * p, **branch_kwargs)
    trunk_layers = list(trunk_layers[:-1]) + [n_out * p]
    self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=activation)

    dummy_x = jnp.zeros((1, *field_shape))
    dummy_t = jnp.zeros((1,))

    def _init_one(key):
      k1, k2 = random.split(key)
      return (self.branch.init(k1, dummy_x, dummy_t)["params"],
              self.trunk_init(rng_key=k2))
    self.params = vmap(_init_one)(random.split(rng_key, n_ens))

    self.optimizer = optax.adam(learning_rate=lr)
    self.opt_state = vmap(self.optimizer.init)(self.params)

    self.loss_don_log = []

  # ---- Core network ----
  def branch_latents(self, params, fields, time):
    B = self.branch.apply({"params": params[0]}, fields, time)
    return B.reshape(fields.shape[0], self.n_out, -1)  # (n_fn, n_out, p)

  def trunk_latents(self, params, y):
    T = self.trunk_apply(params[1], y)
    return T.reshape(*y.shape[:-1], self.n_out, -1)    # (..., n_out, p)

  def predict(self, params, fields, time, y):
    """Per-function points y (n_fn, n_pts, 3) -> (n_fn, n_pts, n_out)."""
    B = self.branch_latents(params, fields, time)
    T = self.trunk_latents(params, y)
    return jnp.einsum("fcp,fncp->fnc", B, T)

  def predict_grid(self, params, fields, time, grid):
    """One shared grid (n_grid, 3) -> (n_fn, n_grid, n_out)."""
    B = self.branch_latents(params, fields, time)
    T = self.trunk_latents(params, grid)
    return jnp.einsum("fcp,gcp->fgc", B, T)

  # ---- Loss ----
  def loss_don(self, params, batch):
    fields, time, y, s = batch
    pred = self.predict(params, fields, time, y)
    return jnp.mean(quantile_loss(self.alpha, s, pred))

  @partial(jit, static_argnums=(0,))
  def _step_all(self, params, opt_state, batch):
    def step_one(params, opt_state, batch):
      loss, g = jax.value_and_grad(self.loss_don)(params, batch)
      updates, opt_state = self.optimizer.update(g, opt_state)
      return optax.apply_updates(params, updates), opt_state, loss
    return vmap(step_one)(params, opt_state, batch)

  # ---- Training loop ----
  def train(self, sampler, nIter, log_every=20):
    """`sampler.next_batch()` returns a batch with a leading n_ens axis."""
    pbar = trange(nIter, desc="Training Ensemble")
    for it in pbar:
      batch = sampler.next_batch()
      self.params, self.opt_state, losses = self._step_all(
          self.params, self.opt_state, batch
      )
      if it % log_every == 0:
        # Mean over members of the loss on the batch before this step.
        train_loss = float(jnp.mean(losses))
        self.loss_don_log.append(train_loss)
        pbar.set_postfix({"train_loss": train_loss})


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
