"""SFCQR calibration of the DeepONet/Burgers pool, as a function of ensemble size.

Same algorithm as the UNet pipeline (`calibrate_sfcqr.py`), applied to the
pool built by `train_burgers_pool.py`. For every (alpha, lr) and each
requested ensemble size it draws `n_repeats` random lo/hi member subsets and,
on the held-out calibration functions (`idx_cp` in `split.npz`):
  1. consensus bounds qbar_lo/qbar_hi = mean over the subset's members,
     scale sigma = max(qbar_hi - qbar_lo, 0) + delta (+ member std, variant B);
  2. per-function score = ceil((1 - alpha) m)-th smallest normalised residual;
  3. Q_p = ceil((n_cal + 1)(1 - beta))-th smallest function score.

The member indices are saved with each Q_p so `analysis_burgers_pool.py`
replays exactly the subset that produced it.

Output: <workdir>/alpha_<a>/lr_<lr>/qp_by_ensemble_size.npz with
  n_ens_<k>         [n_repeats]     Q_p per repeat
  n_ens_<k>_lo_idx  [n_repeats, k]  lo member indices per repeat
  n_ens_<k>_hi_idx  [n_repeats, k]  hi member indices per repeat

Usage:
  python calibrate_burgers_pool.py --config=configs/burgers_config.py:burgers_sfcqr \
      --n_ens_sub_list=1,2,4,8,16 --n_repeats=20
"""

import os
import sys

from absl import app, flags
from jax import config as jax_config
import numpy as np
from ml_collections import config_flags

sys.path.append("./")

from utils.burgers_pool_utils import (
    discover_batch_dirs,
    generate_function_data,
    load_split,
    make_pool_predictor,
    predict_pool,
    quantile_dir,
    restore_pool_params,
    sfcqr_consensus,
    sfcqr_delta,
    sfcqr_function_scores,
    sfcqr_threshold,
)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/burgers_config.py")
flags.DEFINE_string("n_ens_sub_list", "1,2,4,8", "Comma-separated ensemble sizes.")
flags.DEFINE_integer("n_repeats", 20, "Random member subsets per ensemble size.")
flags.DEFINE_integer("subsample_seed", 0, "Seed for drawing member subsets.")
flags.DEFINE_integer("predict_chunk_size", 8, "Pool members predicted at once.")

QP_FILENAME = "qp_by_ensemble_size.npz"


def main(_):
  cfg = FLAGS.config
  jax_config.update("jax_enable_x64", True)
  workdir_root = os.path.abspath(cfg.workdir)
  n_ens_sub_list = [int(x) for x in FLAGS.n_ens_sub_list.split(",")]
  rng = np.random.default_rng(FLAGS.subsample_seed)

  u_all, y_all, s_all = generate_function_data(cfg)
  idx_train, idx_cp = load_split(workdir_root)
  delta = sfcqr_delta(s_all, idx_train)

  n_cal, P = len(idx_cp), cfg.P_train
  u_cp = u_all[idx_cp].reshape(n_cal * P, -1)
  y_cp = y_all[idx_cp].reshape(n_cal * P, -1)
  s_cp = np.asarray(s_all[idx_cp]).reshape(n_cal, P, 1)

  predictor = make_pool_predictor(cfg)

  for alpha in cfg.alpha_list:
    for lr in cfg.lr_list:
      preds = {}
      for label in ("lo", "hi"):
        dirs = discover_batch_dirs(quantile_dir(workdir_root, alpha, lr, label))
        if not dirs:
          break
        params = restore_pool_params(cfg, dirs, lr)
        p = predict_pool(predictor, params, u_cp, y_cp, FLAGS.predict_chunk_size)
        preds[label] = p.reshape(p.shape[0], n_cal, P, 1)
      if len(preds) < 2:
        print(f"Skipping alpha={alpha} lr={lr}: no trained batches")
        continue

      n_lo, n_hi = preds["lo"].shape[0], preds["hi"].shape[0]
      print(f"alpha={alpha} lr={lr}: pool of {n_lo} lo / {n_hi} hi members")

      out = {}
      for k in n_ens_sub_list:
        if k > min(n_lo, n_hi):
          print(f"  skipping n_ens_sub={k} > pool size {min(n_lo, n_hi)}")
          continue
        qp, lo_idx, hi_idx = [], [], []
        for _ in range(FLAGS.n_repeats):
          li = rng.choice(n_lo, size=k, replace=False)
          hi = rng.choice(n_hi, size=k, replace=False)
          qbar_lo, qbar_hi, sigma = sfcqr_consensus(
              preds["lo"][li], preds["hi"][hi], delta, cfg.variant
          )
          S = sfcqr_function_scores(qbar_lo, qbar_hi, sigma, s_cp, alpha)
          qp.append(sfcqr_threshold(S, cfg.beta))
          lo_idx.append(li)
          hi_idx.append(hi)
        qp = np.array(qp)
        out[f"n_ens_{k}"] = qp
        out[f"n_ens_{k}_lo_idx"] = np.stack(lo_idx)
        out[f"n_ens_{k}_hi_idx"] = np.stack(hi_idx)
        print(f"  n_ens_sub={k}: Q_p mean={qp.mean():.4f} std={qp.std():.4f}")

      save_path = os.path.join(workdir_root, f"alpha_{alpha}", f"lr_{lr}", QP_FILENAME)
      np.savez(save_path, **out)
      print(f"  saved {save_path}")


if __name__ == "__main__":
  app.run(main)
