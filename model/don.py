import jax.numpy as jnp
from jax.nn import relu, silu
from jax import random, grad, vmap, jit, lax
import optax

from tqdm import trange
from functools import partial


# Define the neural net
def MLP(layers, activation):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = jnp.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = jnp.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = jnp.dot(inputs, W) + b
      return outputs
  return init, apply

# Quantile loss function
def quantile_loss(alpha, y_true, y_pred):
    e = y_true - y_pred
    return jnp.maximum(alpha * e, (alpha - 1.0) * e)


class DeepONetEnsemble:
    """
    An ensemble of identical DeepONet models trained in parallel
    using vmap.  All ensemble members share architecture but have
    independent parameters and optimizer states.

    With n_out > 1 the branch and trunk each output n_out * p latents, split
    into n_out groups of p, and output c is the dot product of group c
    ("split both"). n_out = 1 is the usual scalar DeepONet.
    """
    def __init__(self, n_ens, branch_layers, trunk_layers, lr,
                 loss_type="quantile", huber_delta=0.4, alpha=0.5,
                 activation=jnp.tanh, rng_key=random.PRNGKey(0), n_out=1):

        self.n_ens      = n_ens
        self.n_out      = n_out
        self.loss_type  = loss_type
        self.huber_delta= huber_delta
        self.alpha      = alpha

        # Last layer of both nets is p per output
        branch_layers = list(branch_layers[:-1]) + [n_out * branch_layers[-1]]
        trunk_layers  = list(trunk_layers[:-1])  + [n_out * trunk_layers[-1]]

        # Build networks
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=activation)
        self.trunk_init,  self.trunk_apply  = MLP(trunk_layers,  activation=activation)

        # Initialise parameters for all ensemble members
        keys = random.split(rng_key, n_ens)
        def _init_one(key):
            k1,k2 = random.split(key)
            return (self.branch_init(rng_key=k1), self.trunk_init(rng_key=k2))
        self.params = vmap(_init_one)(keys)

        # Optimiser
        # self.optimizer  = optax.adam(optax.exponential_decay(1e-3,2000,0.9))
        self.optimizer  = optax.adam(learning_rate=lr)
        self.opt_state  = vmap(self.optimizer.init)(self.params)
        
        # Logs
        self.loss_don_log  = []
        self.loss_test_log = []

    # ---- Core network ----
    def operator_net(self, params, f, z):
        branch_params, trunk_params = params
        if z.ndim==1: z = z[None,:]
        B = self.branch_apply(branch_params, f).reshape(self.n_out, -1)
        T = self.trunk_apply(trunk_params,  z).reshape(self.n_out, -1)
        return jnp.sum(B*T, axis=-1)  # (n_out,)

    def predict_u(self, params, f_star, z_star):
        return vmap(self.operator_net, (None,0,0))(params, f_star, z_star).reshape(-1, self.n_out)

    def get_state(self):
        return {
            'params': self.params,
            'opt_state': self.opt_state,
            'loss_log': jnp.array(self.loss_don_log),
            'alpha': self.alpha,
            'n_ens': self.n_ens
        }
    
    # ---- Loss ----
    def loss_don(self, params, batch):
        inputs, outputs = batch
        f,z = inputs
        pred = vmap(self.operator_net,(None,0,0))(params,f,z)  # (batch, n_out)
        y    = outputs.reshape(pred.shape)
        if   self.loss_type=="huber":
            return jnp.mean(optax.losses.huber_loss(pred, y, delta=self.huber_delta))
        elif self.loss_type=="quantile":
            return jnp.mean(quantile_loss(self.alpha, y, pred))
        else:
            return jnp.mean((y-pred)**2)
    
    def ensemble_loss(self, params_all, batch):

        (f_all, z_all), y_all = batch
        branch_all, trunk_all = params_all

        # ---- CASE 1: Training (data has ensemble axis) ----
        if f_all.ndim > 2:  
            # shape: (n_ens, batch_size, ...)
            def _loss_one(branch_p, trunk_p, f, z, y):
                return self.loss_don((branch_p, trunk_p),
                                    ((f, z), y))

            losses = vmap(_loss_one)(
                branch_all,
                trunk_all,
                f_all,
                z_all,
                y_all
            )

        # ---- CASE 2: Test (shared dataset) ----
        else:
            # shape: (total_points, ...)
            def _loss_one(branch_p, trunk_p):
                return self.loss_don((branch_p, trunk_p),
                                    ((f_all, z_all), y_all))

            losses = vmap(_loss_one)(
                branch_all,
                trunk_all
            )

        return jnp.mean(losses)

    # ---- JIT update for one model ----
    @partial(jit, static_argnums=(0,))
    def _step_single(self, params, opt_state, batch):
        g = grad(self.loss_don)(params, batch)
        updates, opt_state = self.optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # ---- Training loop ----
    def train(self, batched_dataset, nIter, test_data=None, log_every=20):
        pbar = trange(nIter, desc="Training Ensemble")
        for it in pbar:
            # Get one independent batch per model
            batch = batched_dataset.next_batch()  # ((f,z), y), shape (n_ens, batch, ...)

            # step_ensemble: each model gets its own batch
            step_all = vmap(self._step_single, in_axes=(0,0,0))
            self.params, self.opt_state = step_all(self.params, self.opt_state, batch)
            
            # =========================
            # Logging (sparse)
            # =========================
            if it % log_every == 0:

                # ---- Training loss on current batch ----
                train_loss = self.ensemble_loss(self.params, batch)
                self.loss_don_log.append(float(train_loss))

                # ---- Test loss ----
                if test_data is not None:
                    test_batch = ((test_data[0], test_data[1]), test_data[2])
                    test_loss  = self.ensemble_loss(self.params, test_batch)
                    self.loss_test_log.append(float(test_loss))

                pbar.set_postfix({"train_loss": float(train_loss)})

