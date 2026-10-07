from jax import random, jit, lax, vmap
import jax.numpy as jnp

from functools import partial


def bateman_solution(x, t, kappa):
    """Bateman solution of viscous Burgers equation for a shock between f- and f+."""
    return 2 / (1 + jnp.exp( (x - t) / (kappa)))


def solve_burgers_bateman(cfg, kappa, test=False):
    """Compute the Bateman solution on a space-time grid."""
    if(test):
        x = jnp.linspace(cfg.grid_data.xmin, cfg.grid_data.xmax, cfg.P_test)
        t = jnp.linspace(cfg.grid_data.tmin, cfg.grid_data.tmax, cfg.P_test)
    else:
        x = jnp.linspace(cfg.grid_data.xmin, cfg.grid_data.xmax, cfg.grid_data.Nx)
        t = jnp.linspace(cfg.grid_data.tmin, cfg.grid_data.tmax, cfg.grid_data.Nt)

    # Fill solution array
    if(test):
        u = jnp.zeros((cfg.P_test, cfg.P_test))
        for i in range(cfg.P_test):
            u = u.at[:, i].set(bateman_solution(x, t[i], kappa))
    else:
        u = jnp.zeros((cfg.grid_data.Nx, cfg.grid_data.Nt)) 
        for i in range(cfg.grid_data.Nt):
            u = u.at[:, i].set(bateman_solution(x, t[i], kappa))

    return x, t, u


# Generate training data corresponding to one input sample
def generate_one_training_data(key, cfg, kappa):
    # Analytic Bateman solution
    x, t, s = solve_burgers_bateman(cfg, kappa)

    # Initial condition u(x,0)
    x_ini = jnp.linspace(cfg.grid_data.xmin,cfg.grid_data.xmax,cfg.model.branch_layers[0])
    u0 = bateman_solution(x_ini, 0.0, kappa)

    # Input data repeated P times
    u_train = jnp.tile(u0, (cfg.P_train, 1))

    # Random subsamples of space-time points
    subkeys = random.split(key, 3)
    x_2_idx = random.choice(subkeys[0], jnp.arange(cfg.grid_data.Nx), shape=(cfg.P_train, 1))
    t_2_idx = random.choice(subkeys[1], jnp.arange(cfg.grid_data.Nt), shape=(cfg.P_train, 1))

    x_2 = x[x_2_idx]
    t_2 = t[t_2_idx]
    y_train_don = jnp.hstack((x_2, t_2))

    # Corresponding solution values
    s_true = s[x_2_idx, t_2_idx]
    if cfg.noise_scale > 0.0:
        noise = cfg.noise_scale * random.laplace(subkeys[2], shape=s_true.shape)
        s_train_don = s_true + noise
    else:
        s_train_don = s_true
    return u_train, y_train_don, s_train_don


# Generate test data corresponding to one input sample
def generate_one_test_data(cfg, kappa):
    x, t, s = solve_burgers_bateman(cfg, kappa, test=True)

    XX, TT = jnp.meshgrid(x, t, indexing="ij")

    # Initial condition
    x_ini = jnp.linspace(cfg.grid_data.xmin, cfg.grid_data.xmax, cfg.model.branch_layers[0])
    u0 = bateman_solution(x_ini, 0.0, kappa)
    u_test = jnp.tile(u0, (cfg.P_test**2, 1))

    y_test = jnp.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_flat = s.flatten()
    if cfg.noise_scale > 0.0:
        key = random.PRNGKey(123)
        noise = cfg.noise_scale * random.laplace(key, shape=s_flat.shape)
        s_test = s_flat + noise
    else:
        s_test = s_flat

    return u_test, y_test, s_test


# Generate test data for plotting
def generate_test_data_visualization(cfg, kappa):
    u_test, y_test, s_test = generate_one_test_data(cfg, kappa)

    u_test = jnp.float32(u_test.reshape(-1, u_test.shape[-1]))
    y_test = jnp.float32(y_test.reshape(-1, y_test.shape[-1]))
    s_test = jnp.float32(s_test.reshape(-1, 1))

    return u_test, y_test, s_test


# Dataset generator class
class DataGenerator:
    def __init__(self, f, z, labels, batch_size=64, rng_key=random.PRNGKey(1234)):
        self.f = f
        self.z = z
        self.labels = labels
        self.N = f.shape[0]
        self.batch_size = batch_size
        self.key = rng_key
        self.indices = jnp.arange(self.N)
        self.index = 0

    def __iter__(self):
        # Shuffle indices at the start of each epoch
        self.key, subkey = random.split(self.key)
        self.indices = random.permutation(subkey, self.N)
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.N:
            # Epoch finished, reshuffle for next epoch
            self.key, subkey = random.split(self.key)
            self.indices = random.permutation(subkey, self.N)
            self.index = 0

        # Get next batch indices
        batch_idx = self.indices[self.index:self.index + self.batch_size]
        f_batch = self.f[batch_idx, :]
        z_batch = self.z[batch_idx, :]
        labels_batch = self.labels[batch_idx, :]
        self.index += self.batch_size

        return (f_batch, z_batch), labels_batch


class BatchedDataGenerator:
    def __init__(self, base_generator: DataGenerator, n_ens: int):
        self.base = base_generator
        self.n_ens = n_ens

    def next_batch(self):
        """
        Return a tuple of inputs and outputs:
          inputs : (f, z) each with leading axis n_ens
          outputs: labels with leading axis n_ens
        """
        f_list, z_list, y_list = [], [], []
        for _ in range(self.n_ens):
            (f, z), y = next(self.base)
            f_list.append(f)
            z_list.append(z)
            y_list.append(y)
        f = jnp.stack(f_list)   # (n_ens, batch, ...)
        z = jnp.stack(z_list)
        y = jnp.stack(y_list)
        return (f, z), y
