import jax
import jax.numpy as jnp
import numpy as np
import apebench

# Enable 64-bit precision
# jax.config.update("jax_enable_x64", True)


def generate_ks3d_da_data(
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
    
    # Force off-device, force numpy host arrays
    ground_truth = jax.device_get(ground_truth)
    observations = jax.device_get(observations)
    times = jax.device_get(times)

    return np.asarray(ground_truth), np.asarray(observations), np.asarray(times)