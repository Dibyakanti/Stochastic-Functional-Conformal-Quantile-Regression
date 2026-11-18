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
    fields are used raw (no normalisation), in the file's units.

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

    # Singleton z axis so the 3D U-Net / SFCQR code can be reused unchanged.
    ground_truth = np.ascontiguousarray(ground_truth[:, :, :, None, :])

    print("Adding observation noise...")
    key = jax.random.PRNGKey(seed)
    noise = jax.random.laplace(key, shape=ground_truth.shape) * noise_std
    observations = ground_truth + np.asarray(jax.device_get(noise), dtype=np.float32)

    times = np.arange(time_steps + 1) * dt

    return ground_truth, observations, times


def load_challenge2_2_da_data(
    data_dir: str = "./dataset",
    case: str = "a25/f0p2",
    time_steps: int = 400,
    t_start: int = 0,
    crop_shape: tuple[int, int] = (56, 56),
    noise_std: float = 0.0,
    seed: int = 42,
):
    """
    Loads one pitching-airfoil trajectory of the Challenge 2.2 data, in the
    same layout as `load_challenge_da_data`.

    The training file holds 16 separate trajectories, `a<mean angle>/f<freq>`
    (e.g. "a25/f0p2"), each (800, 60, 60) per component on a uniform grid.
    The pipeline assumes one continuous trajectory, so a single `case` is
    used. We take `time_steps + 1` consecutive frames from `t_start`, centre-
    crop to `crop_shape` (keeping the airfoil centred) so both spatial dims
    divide the U-Net downsampling factor. Fields are used raw (no
    normalisation). The
    pitch angle (`alpha`, `alphadot`) and airfoil coordinates are not used.

    Returns:
        ground_truth:  (time_steps+1, X, Y, 1, 2)  — singleton z axis
        observations:  (time_steps+1, X, Y, 1, 2)  — ground truth + Laplace noise
        times:         (time_steps+1,)
    """
    with h5py.File(os.path.join(data_dir, "Challenge2_2_parameters.h5"), "r") as f:
        dt = float(f["dt"][()].squeeze())

    t_end = t_start + time_steps + 1
    print(f"Loading Challenge 2.2 case {case} frames [{t_start}, {t_end}) "
          f"cropped to {crop_shape[0]}x{crop_shape[1]}...")
    with h5py.File(os.path.join(data_dir, "Challenge2_2_train.h5"), "r") as f:
        g = f[case]
        if t_end > g["ux"].shape[0]:
            raise ValueError(
                f"Requested frames up to {t_end}, but only {g['ux'].shape[0]} exist."
            )
        ux = g["ux"][t_start:t_end].astype(np.float32)
        uy = g["uy"][t_start:t_end].astype(np.float32)

    ground_truth = np.stack([ux, uy], axis=-1)  # (T, X, Y, C)
    x0 = (ground_truth.shape[1] - crop_shape[0]) // 2
    y0 = (ground_truth.shape[2] - crop_shape[1]) // 2
    ground_truth = ground_truth[:, x0 : x0 + crop_shape[0], y0 : y0 + crop_shape[1]]

    # Singleton z axis so the 3D U-Net / SFCQR code can be reused unchanged.
    ground_truth = np.ascontiguousarray(ground_truth[:, :, :, None, :])

    print("Adding observation noise...")
    key = jax.random.PRNGKey(seed)
    noise = jax.random.laplace(key, shape=ground_truth.shape) * noise_std
    observations = ground_truth + np.asarray(jax.device_get(noise), dtype=np.float32)

    times = np.arange(time_steps + 1) * dt

    return ground_truth, observations, times
