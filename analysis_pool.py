"""Rollout coverage analysis over the pool built by `train_ensemble_pool.py`
and calibrated by `calibrate_sfcqr.py`.

This mirrors `analysis_sfcqr_claude.py`'s rollout coverage/interval-size
evaluation, but instead of one fixed `n_ens` ensemble it replays, for every
`n_ens_sub` and every calibration repeat, the *exact same* subsample of pool
members that `calibrate_sfcqr.py` used to compute that repeat's `Q_p`
(`n_ens_{k}_lo_idx` / `n_ens_{k}_hi_idx` in `qp_by_ensemble_size.npz`). Using
the same members for both calibration and evaluation is required -- pairing a
`Q_p` from one subset with predictions from a different subset invalidates
the SFCQR guarantee.

Because each `n_ens_sub` now contributes `n_repeats` coverage/interval-size
points instead of one, `plot_line` (which uses seaborn's `lineplot`) will
automatically aggregate repeated x-values into a mean line with a shaded
error band -- no plotting code changes needed.

Usage (example):
  python analysis_pool.py --config configs/config.py:sfcqr_smallnn_beltrami_adamw \
      --batch_n_ens 8
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags
from ml_collections import config_flags

sys.path.append("./")

from model.unet3d import UNet3d
from data.beltramiDataGeneration import *
from data.challengeDataLoading import *
from data.ks3dDataGeneration import *
from utils.plot_utils import plot_line

from calibrate_sfcqr import (
    discover_batch_dirs,
    restore_pool_variables,
    load_split,
    _build_chunked_apply,
)

FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "configs/config.py")
# flags.DEFINE_integer(
#     "batch_n_ens", 8, "Number of ensemble members per training batch. Must "
#     "match the value used with `train_ensemble_pool.py` / `calibrate_sfcqr.py`."
# )
# flags.DEFINE_integer(
#     "predict_chunk_size", 4, "Number of ensemble members to run through the "
#     "U-Net at once. Lower this if you hit an out-of-memory error."
# )


def predict_with_dt(unet, variables, obs, dt_input, member_chunk_size=4):
  """Same chunked-over-ensemble-axis prediction as
  `calibrate_sfcqr.predict_pool`, but taking an explicit `dt_input` (rollout
  uses a different dt than calibration). Uses `calibrate_sfcqr`'s cached
  jitted-vmap builder (`_build_chunked_apply`) so the U-Net is compiled once
  and reused across every unroll step / repeat / n_ens_sub -- passing `obs`
  as a real argument rather than closing over it. Building a fresh
  `jax.jit(jax.vmap(...))` per call, or closing over `obs`, forces a full
  recompile every call, which is what made earlier runs take hours instead
  of minutes. In practice `n_members` here is just `n_ens_sub` (small), so
  this is usually a single chunk / a plain `vmap`."""
  params = variables["params"]
  mutables = {k: v for k, v in variables.items() if k != "params"}

  chunked_apply = _build_chunked_apply(unet)

  n_members = jax.tree.leaves(params)[0].shape[0]
  chunk = min(member_chunk_size, n_members)

  outputs = []
  for start in range(0, n_members, chunk):
    end = min(start + chunk, n_members)
    p_chunk = jax.tree.map(lambda x: x[start:end], params)
    m_chunk = jax.tree.map(lambda x: x[start:end], mutables)
    outputs.append(chunked_apply(p_chunk, m_chunk, obs, dt_input))

  return jnp.concatenate(outputs, axis=0)


def plot_functional_coverage(s, func_cov, alpha, beta, channel, folder_name, filename):
  df = pd.DataFrame({"s": s, "cov": func_cov, "alpha": [f"{a:.3f}" for a in alpha]})
  plt.figure(figsize=(8, 6))
  sns.lineplot(data=df, x="s", y="cov", hue="alpha", marker="o")
  plt.axhline(1 - beta, color="black", linestyle=":", linewidth=1, label=r"$1-\beta$")
  plt.xlabel(r"$s$")
  plt.ylabel(f"Functional Coverage Fraction ({channel})")
  plt.title(rf"Functional coverage vs $s$ (target $1-\beta$ = {1 - beta:.2f})")
  plt.legend(title=r"$\alpha$")
  os.makedirs(folder_name, exist_ok=True)
  plt.savefig(f"{folder_name}/{filename}.png", bbox_inches="tight")
  plt.close()


def select_members(variables, member_idx):
  return jax.tree.map(lambda x: x[member_idx], variables)


def run_analysis(cfg):
  workdir_root = os.path.abspath(cfg.workdir)
  batch_n_ens = FLAGS.batch_n_ens

  if cfg.type == "beltrami":
    out_dim = 4
    gt, obs, coords, times = generate_beltrami_da_data(
        spatial_res=cfg.res,
        time_steps=cfg.time_steps,
        dt=cfg.dt,
        a=1.0,
        d=1.0,
        noise_std=cfg.noise_std,
    )
  elif cfg.type == "ks3d":
    out_dim = 1
    gt, obs, times = generate_ks3d_da_data(
        spatial_res=cfg.res, time_steps=cfg.time_steps, noise_std=cfg.noise_std
    )
    # Despite the docstring in `generate_ks3d_da_data`, it does NOT squeeze
    # out the channel axis -- `jnp.moveaxis(raw[0], 1, -1)` moves it to the
    # end, so `gt`/`obs` already come back as (T, X, Y, Z, 1), matching the
    # U-Net's trailing out_dim=1 channel axis. No reshape needed here.

  elif cfg.type == "challenge2_1":
    out_dim = 2
    gt, obs, times = load_challenge_da_data(
        data_dir=cfg.data_dir, time_steps=cfg.time_steps, noise_std=cfg.noise_std
    )
  elif cfg.type == "challenge2_2":
    out_dim = 2
    gt, obs, times = load_challenge2_2_da_data(
        data_dir=cfg.data_dir, case=cfg.case, time_steps=cfg.time_steps,
        noise_std=cfg.noise_std,
    )

  unet = UNet3d(
      out_channels=out_dim,
      time_cond=True,
      num_channels=cfg.model.num_channels,
      use_spatial_attention=cfg.model.use_spatial_attention,
      downsample_ratio=cfg.model.downsample_ratio,
  )
  input_shape = (cfg.batch_size, *gt.shape[1:])

  train_idx, _cal_idx = load_split(workdir_root)
  y_train_true = gt[train_idx + 2]
  delta = 1e-6 * jnp.std(y_train_true.reshape(-1, out_dim), axis=0)
  delta = delta.reshape(1, 1, 1, 1, out_dim)

  coverage_list, s_hmap_list, alpha_hmap_list, interval_size_list = [], [], [], []
  func_coverage_list = []

  for alpha in cfg.alpha_list:
    experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}")

    qp_path = os.path.join(experiment_dir, "qp_by_ensemble_size.npz")
    if not os.path.exists(qp_path):
      print(f"Skipping alpha={alpha}: {qp_path} not found (run calibrate_sfcqr.py first)")
      continue
    qp_data = np.load(qp_path)

    tau_lo, tau_hi = alpha / 2, 1 - alpha / 2

    lo_batch_dirs = discover_batch_dirs(os.path.join(experiment_dir, "lo"))
    hi_batch_dirs = discover_batch_dirs(os.path.join(experiment_dir, "hi"))
    pool_lo_vars = restore_pool_variables(
        lo_batch_dirs, unet, input_shape, tau_lo, batch_n_ens, cfg
    )
    pool_hi_vars = restore_pool_variables(
        hi_batch_dirs, unet, input_shape, tau_hi, batch_n_ens, cfg
    )

    n_ens_sub_list = sorted(
        int(k.split("_")[-1])
        for k in qp_data.files
        if k.startswith("n_ens_") and not k.endswith("_idx")
    )

    for n_ens_sub in n_ens_sub_list:
      qp_draws = qp_data[f"n_ens_{n_ens_sub}"]        # [n_repeats, n_channels]
      lo_idx_draws = qp_data[f"n_ens_{n_ens_sub}_lo_idx"]  # [n_repeats, n_ens_sub]
      hi_idx_draws = qp_data[f"n_ens_{n_ens_sub}_hi_idx"]  # [n_repeats, n_ens_sub]
      n_repeats = qp_draws.shape[0]

      print(f"\n>>> Analyzing Alpha: {alpha} | N_ens_sub: {n_ens_sub} | {n_repeats} repeats")

      for r in range(n_repeats):
        Q_p = jnp.asarray(qp_draws[r])
        sub_lo_vars = select_members(pool_lo_vars, lo_idx_draws[r])
        sub_hi_vars = select_members(pool_hi_vars, hi_idx_draws[r])

        start_idx = 2
        all_preds_lo, all_preds_hi, actual_truth = [], [], []

        for i in range(cfg.num_unroll_steps):
          current_state = obs[start_idx + (2 * i) : start_idx + (2 * i) + 1]
          batch_size = current_state.shape[0]
          dt_val = times[2] - times[0]
          dt_input = jnp.full((batch_size,), dt_val)

          y_reps_lo = predict_with_dt(
              unet, sub_lo_vars, current_state, dt_input,
              member_chunk_size=FLAGS.predict_chunk_size,
          )
          y_reps_hi = predict_with_dt(
              unet, sub_hi_vars, current_state, dt_input,
              member_chunk_size=FLAGS.predict_chunk_size,
          )

          qbar_lo = jnp.mean(y_reps_lo, axis=0)
          qbar_hi = jnp.mean(y_reps_hi, axis=0)

          w = qbar_hi - qbar_lo
          std_lo = jnp.std(y_reps_lo, axis=0)
          std_hi = jnp.std(y_reps_hi, axis=0)
          d = 0.5 * (std_lo + std_hi)

          if cfg.variant == "A":
            sigma = jnp.maximum(w, 0.0) + delta
          elif cfg.variant == "B":
            sigma = jnp.maximum(w, 0.0) + d + delta
          else:
            raise ValueError(f"Unknown scale variant: {cfg.variant}")

          p_lo = qbar_lo - Q_p * sigma
          p_hi = qbar_hi + Q_p * sigma

          all_preds_lo.append(p_lo)
          all_preds_hi.append(p_hi)
          actual_truth.append(gt[start_idx + (2 * i) + 2])

        preds_lo = jnp.concatenate(all_preds_lo, axis=0)
        preds_hi = jnp.concatenate(all_preds_hi, axis=0)
        truth = jnp.stack(actual_truth)

        is_covered = (truth >= preds_lo) & (truth <= preds_hi)
        interval_widths = preds_hi - preds_lo

        # Both types now have a trailing channel axis (size 4 for beltrami,
        # 1 for ks3d) -- reduce over every other axis (time + spatial) to
        # get one scalar per channel.
        reduce_axes = tuple(range(is_covered.ndim - 1))
        coverage = jnp.mean(is_covered, axis=reduce_axes)
        avg_interval_sizes = jnp.mean(interval_widths, axis=reduce_axes)

        # Functional coverage: fraction of rollout steps (one field each) with
        # at least ceil((1 - alpha) m) of their m grid points covered
        # (SFCQR's target, 1 - beta).
        n_steps, n_channels = is_covered.shape[0], is_covered.shape[-1]
        per_field = is_covered.reshape(n_steps, -1, n_channels)
        k_pts = int(np.ceil((1 - alpha) * per_field.shape[1]))
        func_coverage = jnp.mean(per_field.sum(axis=1) >= k_pts, axis=0)

        coverage_list.append(np.array(coverage))
        s_hmap_list.append(n_ens_sub)
        alpha_hmap_list.append(alpha)
        interval_size_list.append(np.array(avg_interval_sizes))
        func_coverage_list.append(np.array(func_coverage))

  coverages = np.array(coverage_list)
  intervals = np.array(interval_size_list)
  func_coverages = np.array(func_coverage_list)
  s_vals = np.array(s_hmap_list)
  alpha_vals = np.array(alpha_hmap_list)
  target_coverage = 1.0 - alpha_vals

  channel_names = {
      "beltrami": ["u", "v", "w", "p"],
      "challenge2_1": ["ux", "uy"],
      "challenge2_2": ["ux", "uy"],
  }.get(cfg.type, ["u"])
  lr_suffix = f"lr{cfg.learning_rate}"
  folder_name = os.path.join(workdir_root, "analysis_pool_plots")
  os.makedirs(folder_name, exist_ok=True)

  offset = 0

  for i, name in enumerate(channel_names):
    chan_coverage = coverages[:, i]
    chan_intervals = intervals[:, i]

    print(f"\nFinal Metrics for Channel {name}:")
    print(f"Mean Coverage: {np.mean(chan_coverage) * 100:.2f}%")
    for alpha in np.unique(alpha_vals):
      for s in np.unique(s_vals):
        sel = (alpha_vals == alpha) & (s_vals == s)
        if not sel.any():
          continue
        print(
            f"alpha={alpha} s={s}: pointwise {coverages[sel, i].mean():.4f} +- "
            f"{coverages[sel, i].std():.4f} (target {1 - alpha:.3f}), functional "
            f"{func_coverages[sel, i].mean():.4f} +- {func_coverages[sel, i].std():.4f} "
            f"(target {1 - cfg.beta:.3f})"
        )

    filename = f"SCQR_Coverage_lineplot_{name}_{lr_suffix}"
    plot_line(
        x_list=s_vals,
        x_label=r"$s$",
        y_list=chan_coverage,
        y_label=f"Coverage Fraction ({name})",
        line_list=target_coverage,
        line_label=r"$1 - \alpha$",
        folder_name=folder_name,
        filename=filename,
        draw_line=True,
        offset=offset,
    )

    filename = f"SCQR_Interval_lineplot_{name}_{lr_suffix}"
    plot_line(
        x_list=s_vals,
        x_label=r"$s$",
        y_list=chan_intervals,
        y_label=f"Interval Size ({name})",
        line_list=target_coverage,
        line_label=r"$1 - \alpha$",
        folder_name=folder_name,
        filename=filename,
        y_scale = "log",
    )

    plot_functional_coverage(
        s_vals, func_coverages[:, i], alpha_vals, cfg.beta, name, folder_name,
        f"SCQR_FunctionalCoverage_lineplot_{name}_{lr_suffix}",
    )

    # filename = f"SCQR_Interval_Coverage_lineplot_{name}_{lr_suffix}"
    # plot_line(
    #     x_list=chan_coverage,
    #     x_label="Coverage Fraction",
    #     y_list=chan_intervals,
    #     y_label="Interval Size",
    #     line_list=s_vals,
    #     line_label=r"$s$",
    #     folder_name=folder_name,
    #     filename=filename,
    #     shaded=True,
    #     draw_c_true=True,
    # )


def main(_):
  run_analysis(FLAGS.config)


if __name__ == "__main__":
  app.run(main)
