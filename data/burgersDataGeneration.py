from jax import random, jit, lax, vmap
import jax.numpy as jnp

from functools import partial


def burgers_solution(x, t, nu):
    """Analytic solution of the d-dimensional viscous Burgers equation

        u_t + u * sum_i du/dx_i - nu * sum_i d^2u/dx_i^2 = 0,
        u(x, t) = 1 / (1 + exp((sum_i x_i - d t / 2) / (2 nu))).

    x has shape (..., d) and t broadcasts against x[..., 0].
    """
    d = x.shape[-1]
    return 1 / (1 + jnp.exp((jnp.sum(x, axis=-1) - d * t / 2) / (2 * nu)))


def sample_points(key, cfg, n):
    """n uniform (x, t) points in [xmin, xmax]^dim x [tmin, tmax], shape (n, dim + 1)."""
    g = cfg.grid_data
    key_x, key_t = random.split(key)
    x = random.uniform(key_x, (n, cfg.dim), minval=g.xmin, maxval=g.xmax)
    t = random.uniform(key_t, (n, 1), minval=g.tmin, maxval=g.tmax)
    return jnp.hstack((x, t))


def sensor_locations(cfg):
    """Fixed branch-input sensors in [xmin, xmax]^dim, shared by every function."""
    m = cfg.model.branch_layers[0]
    return random.uniform(random.PRNGKey(2024), (m, cfg.dim),
                          minval=cfg.grid_data.xmin, maxval=cfg.grid_data.xmax)


def initial_condition(cfg, nu):
    """u(x, 0) at the sensor locations, shape (m,)."""
    return burgers_solution(sensor_locations(cfg), 0.0, nu)


def _add_noise(key, cfg, s):
    if cfg.noise_scale > 0.0:
        return s + cfg.noise_scale * random.laplace(key, shape=s.shape)
    return s


# Generate training data corresponding to one input sample
def generate_one_training_data(key, cfg, nu):
    # Input data repeated P times
    u_train = jnp.tile(initial_condition(cfg, nu), (cfg.P_train, 1))

    # Random space-time points and corresponding solution values
    key_y, key_noise = random.split(key)
    y_train_don = sample_points(key_y, cfg, cfg.P_train)
    s_true = burgers_solution(y_train_don[:, :-1], y_train_don[:, -1], nu)[:, None]
    s_train_don = _add_noise(key_noise, cfg, s_true)
    return u_train, y_train_don, s_train_don


# Generate test data corresponding to one input sample
def generate_one_test_data(cfg, nu):
    """P_test**2 space-time points, the same for every test function."""
    n_pts = cfg.P_test**2
    u_test = jnp.tile(initial_condition(cfg, nu), (n_pts, 1))

    y_test = sample_points(random.PRNGKey(321), cfg, n_pts)
    s_flat = burgers_solution(y_test[:, :-1], y_test[:, -1], nu)
    s_test = _add_noise(random.PRNGKey(123), cfg, s_flat)
    return u_test, y_test, s_test


# Generate test data for plotting
def generate_test_data_visualization(cfg, nu):
    u_test, y_test, s_test = generate_one_test_data(cfg, nu)

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
