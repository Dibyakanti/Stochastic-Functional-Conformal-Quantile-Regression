import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Enable 64-bit precision for accurate physics simulation
jax.config.update("jax_enable_x64", True)

@jax.jit
def beltrami_snapshot(t, grid_x, grid_y, grid_z, a, d):
    """
    Computes the exact solution for Beltrami flow at a specific time t
    over the entire grid (vectorized).
    """
    # Decay terms
    decay_vel = jnp.exp(-d**2 * t)
    decay_pres = jnp.exp(-2 * d**2 * t)

    # Spatial exponentials
    e_ax = jnp.exp(a * grid_x)
    e_ay = jnp.exp(a * grid_y)
    e_az = jnp.exp(a * grid_z)

    # Precompute trigonometric arguments
    # u uses: sin(ay + dz), cos(ax + dy)
    # v uses: sin(az + dx), cos(ay + dz)
    # w uses: sin(ax + dy), cos(az + dx)
    
    arg_xy = a * grid_x + d * grid_y  # ax + dy
    arg_yz = a * grid_y + d * grid_z  # ay + dz
    arg_zx = a * grid_z + d * grid_x  # az + dx

    # --- Velocity Components ---
    # u(x,y,z,t)
    u = -a * (e_ax * jnp.sin(arg_yz) + e_az * jnp.cos(arg_xy)) * decay_vel
    
    # v(x,y,z,t)
    v = -a * (e_ay * jnp.sin(arg_zx) + e_ax * jnp.cos(arg_yz)) * decay_vel
    
    # w(x,y,z,t)
    w = -a * (e_az * jnp.sin(arg_xy) + e_ay * jnp.cos(arg_zx)) * decay_vel

    # --- Pressure Component ---
    # Squared exponential terms
    p_sq_terms = jnp.exp(2*a*grid_x) + jnp.exp(2*a*grid_y) + jnp.exp(2*a*grid_z)
    
    # Cross terms from equation (11)
    # Term 1: 2 sin(ax+dy) cos(az+dx) e^{a(y+z)}
    ct1 = 2 * jnp.sin(arg_xy) * jnp.cos(arg_zx) * jnp.exp(a * (grid_y + grid_z))
    
    # Term 2: 2 sin(ay+dz) cos(ax+dy) e^{a(z+x)}
    ct2 = 2 * jnp.sin(arg_yz) * jnp.cos(arg_xy) * jnp.exp(a * (grid_z + grid_x))
    
    # Term 3: 2 sin(az+dx) cos(ay+dz) e^{a(x+y)}
    ct3 = 2 * jnp.sin(arg_zx) * jnp.cos(arg_yz) * jnp.exp(a * (grid_x + grid_y))

    p = -0.5 * (a**2) * (p_sq_terms + ct1 + ct2 + ct3) * decay_pres

    # Stack channels at the end: Shape (X, Y, Z, 4) -> [u, v, w, p]
    return jnp.stack([u, v, w, p], axis=-1)

def generate_beltrami_da_data(
    spatial_res=32, 
    time_steps=50, 
    dt=0.01, 
    a=1.0, 
    d=1.0, 
    noise_std=0.05,
    seed=42
):
    """
    Generates a 4D dataset for Data Assimilation.
    Returns:
        ground_truth: The exact physics (time, X, Y, Z, 4)
        observations: The ground truth + Gaussian noise (time, X, Y, Z, 4)
        coords: Dictionary of spatial coordinates
        times: Array of time points
    """
    # 1. Define Grid (Domain [-1, 1])
    x = jnp.linspace(-1, 1, spatial_res)
    y = jnp.linspace(-1, 1, spatial_res)
    z = jnp.linspace(-1, 1, spatial_res)
    grid_x, grid_y, grid_z = jnp.meshgrid(x, y, z, indexing='ij')

    # 2. Define Time
    times = jnp.arange(time_steps) * dt

    # 3. Compute Ground Truth (Vectorized over time)
    # We use vmap to apply the snapshot function over the time array
    # in_axes: (0, None, None, None, None, None) means vectorize over 't' (first arg), keep others fixed
    compute_trajectory = jax.vmap(beltrami_snapshot, in_axes=(0, None, None, None, None, None))
    
    print(f"Generating Ground Truth for grid {spatial_res}x{spatial_res}x{spatial_res} over {time_steps} steps...")
    ground_truth = compute_trajectory(times, grid_x, grid_y, grid_z, a, d)
    
    # 4. Generate Noisy Observations
    print("Adding observation noise...")
    key = jax.random.PRNGKey(seed)
    noise = jax.random.laplace(key, shape=ground_truth.shape) * noise_std
    observations = ground_truth + noise

    return ground_truth, observations, {"x": x, "y": y, "z": z}, times


# def beltrami_dataloader(
#     ground_truth: jax.Array, 
#     observations: jax.Array, 
#     times: jax.Array, 
#     batch_size: int, 
#     shuffle: bool = True
# ):
#     """
#     Slices a (time, X, Y, Z, C) trajectory into (x, y) pairs for DA.
#     x: state at t
#     y: state at t+dt
#     """
#     num_time_steps = observations.shape[0]
#     # We have (T-1) possible pairs
#     indices = np.arange(num_time_steps - 1)
    
#     while True:
#         if shuffle:
#             np.random.shuffle(indices)
        
#         for i in range(0, len(indices), batch_size):
#             batch_idx = indices[i:i + batch_size]
#             if len(batch_idx) < batch_size:
#                 continue
                
#             # 'x' is the noisy observation at time t
#             # 'y' is the ground truth (or noisy observation) at time t+dt
#             yield {
#                 "x": observations[batch_idx],
#                 "y": observations[batch_idx + 1],
#                 "time": times[batch_idx]
#             }

# def beltrami_da_dataloader(
#     ground_truth: jax.Array, 
#     observations: jax.Array, 
#     times: jax.Array, 
#     batch_size: int, 
#     mode: str = "train"  # "train" for odd, "eval" for even
# ):
#     num_time_steps = observations.shape[0]
#     # Indices 0 to T-2 (to allow for i+1 target)
#     all_indices = np.arange(num_time_steps - 3)
    
#     if mode == "train":
#         # Keep only odd indices: t=1, 3, 5...
#         indices = all_indices[all_indices % 2 != 0]
#     else:
#         # Keep only even indices starting from t=2: t=2, 4, 6...
#         indices = all_indices[(all_indices % 2 == 0) & (all_indices >= 2)]
    
#     while True:
#         np.random.shuffle(indices)
#         for i in range(0, len(indices), batch_size):
#             batch_idx = indices[i:i + batch_size]
#             if len(batch_idx) < batch_size: continue
#             yield {
#                 "x": observations[batch_idx],
#                 "y": observations[batch_idx + 2],
#                 "time": times[batch_idx + 2] - times[batch_idx]
#             }