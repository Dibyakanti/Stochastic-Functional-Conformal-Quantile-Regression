import os

import h5py
import jax
import numpy as np


def load_challenge_da_data(
    data_dir: str = "./dataset",
    time_steps: int = 400,
    t_start: int = 0,
    crop_shape: tuple[int, int] = (152, 48),
    noise_std: float = 0.0,
    seed: int = 42,
):
    """
    Loads the Challenge 2.1 2D velocity data (ux, uy) for Data Assimilation,
    in the same layout as `generate_ks3d_da_data`.

    The training file holds one continuous trajectory of shape (12800, 154, 51)
    per component, on a uniform grid with x stored in *decreasing* order. We
    take `time_steps + 1` consecutive frames starting at `t_start`, flip x to
    increasing order and crop to `crop_shape` (keeping the low-x / low-y
    corner) so both spatial dims divide the U-Net downsampling factor. The
    field is standardised per channel, so the pinball losses, intervals and
    Q_p all live in units of the channel std.

    Returns:
        ground_truth:  (time_steps+1, X, Y, 1, 2)  — singleton z axis
        observations:  (time_steps+1, X, Y, 1, 2)  — ground truth + Laplace noise
        times:         (time_steps+1,)
    """
    with h5py.File(os.path.join(data_dir, "Challenge2_1_parameters.h5"), "r") as f:
        dt = float(f["dt"][()].squeeze())

    t_end = t_start + time_steps + 1
    nx, ny = crop_shape
    print(f"Loading Challenge 2.1 frames [{t_start}, {t_end}) cropped to {nx}x{ny}...")
    with h5py.File(os.path.join(data_dir, "Challenge2_1_train.h5"), "r") as f:
        if t_end > f["ux"].shape[0]:
            raise ValueError(
                f"Requested frames up to {t_end}, but only {f['ux'].shape[0]} exist."
            )
        ux = f["ux"][t_start:t_end].astype(np.float32)
        uy = f["uy"][t_start:t_end].astype(np.float32)

    # (T, X, Y, C), x flipped to increasing order, then cropped.
    ground_truth = np.stack([ux, uy], axis=-1)[:, ::-1][:, :nx, :ny]

    mean = ground_truth.mean(axis=(0, 1, 2))
    std = ground_truth.std(axis=(0, 1, 2))
    print(f"Per-channel mean {mean}, std {std} (data standardised with these)")
    ground_truth = (ground_truth - mean) / std

    # Singleton z axis so the 3D U-Net / SFCQR code can be reused unchanged.
    ground_truth = np.ascontiguousarray(ground_truth[:, :, :, None, :])

    print("Adding observation noise...")
    key = jax.random.PRNGKey(seed)
    noise = jax.random.laplace(key, shape=ground_truth.shape) * noise_std
    observations = ground_truth + np.asarray(jax.device_get(noise), dtype=np.float32)

    times = np.arange(time_steps + 1) * dt

    return ground_truth, observations, times
