"""Test-set coverage of the SFCQR-calibrated DeepONet/Beltrami pool, with error
bars over the choice of ensemble members.

Uses the same test set as the UNet `analysis_pool.py`: inputs obs[2 + 2i] and
truth gt[4 + 2i] on the whole grid, for i < num_unroll_steps. For each
(lr, alpha, ensemble size, repeat) it replays the exact lo/hi member subsets
saved by `calibrate_beltrami_don_pool.py` and builds, per channel u, v, w, p,
  [qbar_lo - Q_p * sigma, qbar_hi + Q_p * sigma]
(same consensus/scale as calibration). It records:
  - pointwise coverage (target 1 - alpha) and mean interval width;
  - functional coverage: the fraction of test time indices with at least
    ceil((1 - alpha) m) of their m grid points covered (target 1 - beta).
Each ensemble size contributes one point per repeat, so the line plots show a
mean with a shaded error band.

Plots: <workdir>/analysis_pool_plots/
Raw metrics: <workdir>/analysis_pool_metrics_lr_<lr>.npz

Usage:
  python analysis_beltrami_don_pool.py \
      --config=configs/beltrami_don_config.py:beltrami_don_sfcqr
"""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"

from absl import app, flags
from jax import config as jax_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ml_collections import config_flags

sys.path.append("./")

from utils.beltrami_don_pool_utils import (
    branch_fields,
    field_targets,
    load_beltrami_data,
    load_split,
    make_field_predictor,
    predict_fields,
    quantile_dir,
    restore_pool_params,
    sfcqr_delta,
    time_gap,
    test_indices,
)
from utils.burgers_pool_utils import discover_batch_dirs, sfcqr_consensus
from utils.plot_utils import plot_line

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/beltrami_don_config.py")
flags.DEFINE_integer("predict_chunk_size", 8, "Pool members predicted at once.")

QP_FILENAME = "qp_by_ensemble_size.npz"


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


def main(_):
  cfg = FLAGS.config
  jax_config.update("jax_enable_x64", True)
  workdir_root = os.path.abspath(cfg.workdir)
  folder_name = os.path.join(workdir_root, "analysis_pool_plots")

  gt, obs, grid = load_beltrami_data(cfg)
  train_idx, _ = load_split(workdir_root)
  delta = sfcqr_delta(gt, train_idx)
  test_idx = test_indices(cfg)
  s_test = field_targets(gt, test_idx)
  f_test = branch_fields(obs, test_idx)
  t_test = time_gap(cfg, len(test_idx))
  grid = np.float32(grid)
  m_test = grid.shape[0]

  predictor = make_field_predictor(cfg)

  for lr in cfg.lr_list:
    print(f"\n--- lr={lr} ---")
    rows = {
        "channel": [], "s": [], "alpha": [],
        "coverage": [], "func_coverage": [], "interval": [],
    }

    for alpha in cfg.alpha_list:
      q_path = os.path.join(workdir_root, f"alpha_{alpha}", f"lr_{lr}", QP_FILENAME)
      if not os.path.exists(q_path):
        print(
            f"Skipping alpha={alpha}: {q_path} not found "
            "(run calibrate_beltrami_don_pool.py)"
        )
        continue
      q_data = np.load(q_path)

      # Each member's test predictions are computed once and reused by every
      # subset; [n_pool, n_test, m_test, 4] per quantile.
      preds = {}
      for label in ("lo", "hi"):
        params = restore_pool_params(
            cfg, discover_batch_dirs(quantile_dir(workdir_root, alpha, lr, label)), lr
        )
        preds[label] = predict_fields(
            predictor, params, f_test, t_test, grid, FLAGS.predict_chunk_size
        )

      k_pts = int(np.ceil((1 - alpha) * m_test))
      sizes = sorted(
          int(key.split("_")[-1]) for key in q_data.files if not key.endswith("_idx")
      )
      for k in sizes:
        q_p = q_data[f"n_ens_{k}"]  # [n_repeats, 4]
        lo_idx, hi_idx = q_data[f"n_ens_{k}_lo_idx"], q_data[f"n_ens_{k}_hi_idx"]
        covs, fcovs = [], []
        for r in range(len(q_p)):
          qbar_lo, qbar_hi, sigma = sfcqr_consensus(
              preds["lo"][lo_idx[r]], preds["hi"][hi_idx[r]], delta, cfg.variant
          )
          lo = qbar_lo - q_p[r] * sigma
          hi = qbar_hi + q_p[r] * sigma
          covered = (s_test >= lo) & (s_test <= hi)  # [n_test, m_test, 4]
          cov = covered.mean(axis=(0, 1))
          fcov = np.mean(covered.sum(axis=1) >= k_pts, axis=0)
          width = np.mean(hi - lo, axis=(0, 1))
          covs.append(cov)
          fcovs.append(fcov)
          for c, channel in enumerate(cfg.channels):
            rows["coverage"].append(float(cov[c]))
            rows["func_coverage"].append(float(fcov[c]))
            rows["interval"].append(float(width[c]))
            rows["s"].append(k)
            rows["alpha"].append(alpha)
            rows["channel"].append(channel)
        covs, fcovs = np.stack(covs), np.stack(fcovs)
        for c, channel in enumerate(cfg.channels):
          print(
              f"{channel} alpha={alpha} s={k}: pointwise {covs[:, c].mean():.4f} +- "
              f"{covs[:, c].std():.4f} (target {1 - alpha:.3f}), functional "
              f"{fcovs[:, c].mean():.4f} +- {fcovs[:, c].std():.4f} "
              f"(target {1 - cfg.beta:.3f})"
          )

    if not rows["s"]:
      continue

    r = {key: np.array(v) for key, v in rows.items()}
    np.savez(os.path.join(workdir_root, f"analysis_pool_metrics_lr_{lr}.npz"), **r)

    for channel in cfg.channels:
      sel = r["channel"] == channel
      if not sel.any():
        continue
      s, target = r["s"][sel], 1 - r["alpha"][sel]
      suffix = f"{channel}_lr_{lr}"
      plot_line(s, r"$s$", r["coverage"][sel], f"Coverage Fraction ({channel})",
                target, r"$1 - \alpha$", folder_name,
                f"SFCQR_Coverage_lineplot_{suffix}", draw_line=True)
      plot_line(s, r"$s$", r["interval"][sel], f"Interval Size ({channel})",
                target, r"$1 - \alpha$", folder_name,
                f"SFCQR_Interval_lineplot_{suffix}", y_scale="log")
      plot_functional_coverage(s, r["func_coverage"][sel], r["alpha"][sel],
                               cfg.beta, channel, folder_name,
                               f"SFCQR_FunctionalCoverage_lineplot_{suffix}")
      plt.close("all")


if __name__ == "__main__":
  app.run(main)
