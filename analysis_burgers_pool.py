"""Test-set coverage of the SFCQR-calibrated DeepONet/Burgers pool, with error
bars over the choice of ensemble members.

For each (lr, alpha, ensemble size, repeat) it replays the exact lo/hi member
subsets saved by `calibrate_burgers_pool.py` and builds, on the test functions,
  [qbar_lo - Q_p * sigma, qbar_hi + Q_p * sigma]
(same consensus/scale as calibration). It records:
  - pointwise coverage (target 1 - alpha) and mean interval width;
  - functional coverage: the fraction of test functions with at least
    ceil((1 - alpha) m) of their m points covered (SFCQR's target, 1 - beta).
Each ensemble size contributes one point per repeat, so the line plots show a
mean with a shaded error band.

Plots: <workdir>/analysis_pool_plots/
Raw metrics: <workdir>/analysis_pool_metrics_lr_<lr>.npz

Usage:
  python analysis_burgers_pool.py --config=configs/burgers_config.py:burgers_sfcqr
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

from utils.burgers_pool_utils import (
    discover_batch_dirs,
    generate_function_data,
    generate_test_data,
    load_split,
    make_pool_predictor,
    predict_pool,
    quantile_dir,
    restore_pool_params,
    sfcqr_consensus,
    sfcqr_delta,
)
from utils.plot_utils import plot_line

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/burgers_config.py")
flags.DEFINE_integer("predict_chunk_size", 8, "Pool members predicted at once.")

QP_FILENAME = "qp_by_ensemble_size.npz"


def plot_functional_coverage(s, func_cov, alpha, beta, folder_name, filename):
  df = pd.DataFrame({"s": s, "cov": func_cov, "alpha": [f"{a:.3f}" for a in alpha]})
  plt.figure(figsize=(8, 6))
  sns.lineplot(data=df, x="s", y="cov", hue="alpha", marker="o")
  plt.axhline(1 - beta, color="black", linestyle=":", linewidth=1, label=r"$1-\beta$")
  plt.xlabel(r"$s$")
  plt.ylabel("Functional Coverage Fraction")
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

  # delta must match calibration, which takes it from the training functions.
  _, _, s_all = generate_function_data(cfg)
  idx_train, _ = load_split(workdir_root)
  delta = sfcqr_delta(s_all, idx_train)

  u_test, y_test, s_test = generate_test_data(cfg)
  s_test = np.asarray(s_test)
  m_test = cfg.P_test**2
  predictor = make_pool_predictor(cfg)

  for lr in cfg.lr_list:
    print(f"\n--- lr={lr} ---")
    rows = {"s": [], "alpha": [], "coverage": [], "func_coverage": [], "interval": []}

    for alpha in cfg.alpha_list:
      q_path = os.path.join(workdir_root, f"alpha_{alpha}", f"lr_{lr}", QP_FILENAME)
      if not os.path.exists(q_path):
        print(f"Skipping alpha={alpha}: {q_path} not found (run calibrate_burgers_pool.py)")
        continue
      q_data = np.load(q_path)

      # Each member's test predictions are computed once and reused by every
      # subset; [n_pool, n_test_points, 1] per quantile.
      preds = {}
      for label in ("lo", "hi"):
        params = restore_pool_params(
            cfg, discover_batch_dirs(quantile_dir(workdir_root, alpha, lr, label)), lr
        )
        preds[label] = predict_pool(
            predictor, params, u_test, y_test, FLAGS.predict_chunk_size
        ).astype(np.float32)

      k_pts = int(np.ceil((1 - alpha) * m_test))
      sizes = sorted(
          int(key.split("_")[-1]) for key in q_data.files if not key.endswith("_idx")
      )
      for k in sizes:
        q_p = q_data[f"n_ens_{k}"]
        lo_idx, hi_idx = q_data[f"n_ens_{k}_lo_idx"], q_data[f"n_ens_{k}_hi_idx"]
        covs, fcovs = [], []
        for r in range(len(q_p)):
          qbar_lo, qbar_hi, sigma = sfcqr_consensus(
              preds["lo"][lo_idx[r]], preds["hi"][hi_idx[r]], delta, cfg.variant
          )
          lo = qbar_lo - q_p[r] * sigma
          hi = qbar_hi + q_p[r] * sigma
          covered = (s_test >= lo) & (s_test <= hi)
          per_fn = covered.reshape(cfg.N_test, m_test).sum(axis=1)

          covs.append(float(covered.mean()))
          fcovs.append(float(np.mean(per_fn >= k_pts)))
          rows["coverage"].append(covs[-1])
          rows["func_coverage"].append(fcovs[-1])
          rows["interval"].append(float(np.mean(hi - lo)))
          rows["s"].append(k)
          rows["alpha"].append(alpha)
        print(
            f"alpha={alpha} s={k}: pointwise {np.mean(covs):.4f} +- {np.std(covs):.4f} "
            f"(target {1 - alpha:.3f}), functional {np.mean(fcovs):.4f} +- "
            f"{np.std(fcovs):.4f} (target {1 - cfg.beta:.3f})"
        )

    if not rows["s"]:
      continue

    r = {key: np.array(v) for key, v in rows.items()}
    target = 1 - r["alpha"]
    np.savez(os.path.join(workdir_root, f"analysis_pool_metrics_lr_{lr}.npz"), **r)

    suffix = f"lr_{lr}"
    offset = 1 / (1 + (cfg.N_train / 2 * cfg.P_train))
    plot_line(r["s"], r"$s$", r["coverage"], "Coverage Fraction", target,
              r"$1 - \alpha$", folder_name, f"SFCQR_Coverage_lineplot_{suffix}",
              draw_line=True, offset=offset)
    plot_line(r["s"], r"$s$", target - r["coverage"], r"Distance from $1-\alpha$",
              target, r"$1 - \alpha$", folder_name, f"SFCQR_Distance_lineplot_{suffix}")
    plot_line(r["s"], r"$s$", r["interval"], "Interval Size", target,
              r"$1 - \alpha$", folder_name, f"SFCQR_Interval_lineplot_{suffix}")
    plot_functional_coverage(r["s"], r["func_coverage"], r["alpha"], cfg.beta,
                             folder_name, f"SFCQR_FunctionalCoverage_lineplot_{suffix}")

    # Coverage is the x-axis here, so per-repeat points would zigzag; plot the
    # per-(alpha, s) means instead.
    keys = sorted(set(zip(r["alpha"].tolist(), r["s"].tolist())))
    sel = [(r["alpha"] == a) & (r["s"] == s) for a, s in keys]
    plot_line([r["coverage"][m].mean() for m in sel], "Coverage Fraction",
              [r["interval"][m].mean() for m in sel], "Interval Size",
              [s for _, s in keys], r"$s$", folder_name,
              f"SFCQR_Interval_Coverage_lineplot_{suffix}",
              shaded=True, draw_c_true=True, noise_scale=cfg.noise_scale)
    plt.close("all")


if __name__ == "__main__":
  app.run(main)
