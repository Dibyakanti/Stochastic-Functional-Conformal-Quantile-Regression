import jax
import jax.numpy as jnp
import numpy as np
import apebench

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def generate_da_data(
    spatial_res: int = 16,
    time_steps: int = 50,
    noise_std: float = 0.00,
    seed: int = 42,
    num_samples: int = 1,
):
    """
    Generates a dataset for Data Assimilation using APEBench's 3D
    Kuramoto-Sivashinsky scenario.

    APEBench produces trajectories of shape:
        (num_samples, time_steps+1, num_channels, X, Y, Z)

    We squeeze out the sample and channel dims (both 1 here) to return:
        ground_truth:  (time_steps+1, X, Y, Z)
        observations:  (time_steps+1, X, Y, Z)  — ground truth + Laplace noise
        times:         (time_steps+1,)            — integer step indices

    Note: KS in APEBench operates on a periodic domain with a fixed dt
    determined internally by the ETDRK stepper. There are no explicit (a, d)
    parameters as in Beltrami flow.
    """
    scenario = apebench.scenarios.physical.KuramotoSivashinsky(
        num_spatial_dims=3,
        num_points=spatial_res,
        num_train_samples=num_samples,
        train_temporal_horizon=time_steps,
        domain_extent=10.0,
        train_seed=seed,
        # KS needs a warmup to leave the transient regime
        num_warmup_steps=50,
    )

    print(
        f"Generating KS 3D ground truth: "
        f"grid {spatial_res}^3 over {time_steps} steps "
        f"({num_samples} sample(s))..."
    )
    # Shape: (num_samples, time_steps+1, 1, X, Y, Z)
    raw = scenario.get_train_data()
    ground_truth = jnp.moveaxis(raw[0], 1, -1)

    # # Remove sample and channel dims → (time_steps+1, X, Y, Z)
    # ground_truth = raw[0, :, 0, ...]  # first (and only) sample, channel 0

    print("Adding observation noise...")
    key = jax.random.PRNGKey(seed)
    noise = jax.random.laplace(key, shape=ground_truth.shape) * noise_std
    observations = ground_truth + noise

    # Get the exact dt from the internal stepper
    dt = scenario.get_ref_stepper().dt
    times = jnp.arange(time_steps + 1) * dt

    return ground_truth, observations, times


# def dataloader(
#     ground_truth: jax.Array,
#     observations: jax.Array,
#     times: jax.Array,
#     batch_size: int,
#     shuffle: bool = True,
# ):
#     """
#     Slices a (time, X, Y, Z, C) trajectory into (x, y) pairs for DA.
#       x: noisy observation at step t
#       y: noisy observation at step t+1
#     """
#     num_time_steps = observations.shape[0]
#     indices = np.arange(num_time_steps - 1)

#     while True:
#         if shuffle:
#             np.random.shuffle(indices)

#         for i in range(0, len(indices), batch_size):
#             batch_idx = indices[i : i + batch_size]
#             if len(batch_idx) < batch_size:
#                 continue

#             yield {
#                 "x": observations[batch_idx],       # (B, X, Y, Z)
#                 "y": observations[batch_idx + 1],   # (B, X, Y, Z)
#                 "time": times[batch_idx],
#             }


# def da_dataloader(
#     ground_truth: jax.Array,
#     observations: jax.Array,
#     times: jax.Array,
#     batch_size: int,
#     mode: str = "train",  # "train" for odd indices, "eval" for even
# ):
#     """
#     Odd/even split dataloader for DA with a 2-step prediction target.
#       x: noisy observation at step t
#       y: noisy observation at step t+2
#       time: delta t between x and y (always 2 steps)
#     """
#     num_time_steps = observations.shape[0]
#     all_indices = np.arange(num_time_steps - 3)

#     if mode == "train":
#         indices = all_indices[all_indices % 2 != 0]
#     else:
#         indices = all_indices[(all_indices % 2 == 0) & (all_indices >= 2)]

#     while True:
#         np.random.shuffle(indices)
#         for i in range(0, len(indices), batch_size):
#             batch_idx = indices[i : i + batch_size]
#             if len(batch_idx) < batch_size:
#                 continue

#             yield {
#                 "x": observations[batch_idx],           # (B, X, Y, Z)
#                 "y": observations[batch_idx + 2],       # (B, X, Y, Z)
#                 "time": times[batch_idx + 2] - times[batch_idx],  # always 2
#             }