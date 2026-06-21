import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
import optax
import orbax.checkpoint as ocp
from absl import app, flags
from ml_collections import config_flags
from itertools import product
from PIL import Image
from vape4d import render
from vape4d.utils import diverging_alpha

from model.unet3d import UNet3d
from train.models import QuantileEnsembleDAModel  # Use the Ensemble Model
from train.trainers import BeltramiTrainer
from data.beltramiDataGeneration import *
from data.ks3dDataGeneration import *
from utils.video_utils import *

from train.optimizer import *

# os.environ["JAX_PLATFORM_NAME"] = "cpu"

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "configs/config.py")


def get_ensemble_mean_pred(trainer, obs, dt_input):
    variables = trainer.train_state.model_variables
    
    def single_apply(p):
        return trainer.model.model.apply(
            {"params": p}, obs, time=dt_input, is_training=False
        )
    
    # If only 1 ensemble member, skip vmap entirely
    if trainer.model.n_ens == 1:
        # params has a leading batch dim of 1 — squeeze it out
        params = jax.tree_util.tree_map(lambda x: x[0], variables["params"])
        return trainer.model.model.apply(
            {"params": params}, obs, time=dt_input, is_training=False
        )
    
    ensemble_preds = jax.vmap(single_apply)(variables["params"])
    return jnp.mean(ensemble_preds, axis=0)

FIELD_NAMES = ["u", "v", "w", "p"]
CMAP        = "RdBu_r"
DPI         = 100
MARKER_SIZE = 14   # scatter dot size — increase for coarser grids


RENDER_W     = 16      # vape4d render resolution (pixels)
RENDER_H     = 16
DPI          = 90
VIDEO_FPS    = 4        # frames per second in output MP4

# ── vape4d volume render ──────────────────────────────────────────────────────
 
def render_volume(vol_4d,
                  fps=10,
                  duration=2,
                  cmap_name="magma",
                  width=128,
                  height=128,
                  distance_scale=10):

    n_frames = fps * duration

    cmap = diverging_alpha(plt.get_cmap(cmap_name))

    video = render(
        vol_4d,
        cmap,
        np.linspace(0, 1, n_frames),
        width=width,
        height=height,
        distance_scale=distance_scale,
        background=(0, 0, 0, 0),
    )

    return video

def export_all_videos(lo_vol, hi_vol, gt_vol,
                      field_names,
                      alpha,
                      n_ens,
                      lr,
                      fps=10,
                      duration=2,
                      out_dir="vape4d_videos"):

    os.makedirs(out_dir, exist_ok=True)

    T = len(lo_vol)
    n_fields = len(field_names)

    for f in range(n_fields):

        name = field_names[f]

        # ---- build 4D volumes (T, res, res, res)
        lo = stack_time_series([lo_vol[t][f] for t in range(T)])
        hi = stack_time_series([hi_vol[t][f] for t in range(T)])
        gt = stack_time_series([gt_vol[t][f] for t in range(T)])

        for label, vol in [("lower", lo), ("upper", hi), ("truth", gt)]:

            print(f"rendering {name} | {label}")

            video = render_volume(
                vol,
                fps=fps,
                duration=duration
            )

            fname = f"{name}_{label}_alpha{alpha}_nens{n_ens}_lr{str(lr).replace('.', 'p')}.mp4"
            path = os.path.join(out_dir, fname)

            save_video(video, path, fps=fps)

def run_analysis(cfg):
    workdir_root = os.path.abspath(cfg.workdir)
    
    if cfg.type=="beltrami":
        out_dim = 4
    elif cfg.type=="ks3d":
        out_dim = 1
    
    # Define UNet outside the loops for efficiency
    unet = UNet3d(
        out_channels=out_dim,
        time_cond=True,
        num_channels=cfg.model.num_channels,
        use_spatial_attention=cfg.model.use_spatial_attention,
        downsample_ratio=cfg.model.downsample_ratio
    )

    for alpha, n_ens in product(cfg.alpha_list, cfg.n_ens_list):
        print(f"\n>>> Analyzing Alpha: {alpha} | N_ens: {n_ens}")
        
        # 1. Path Setup for this specific experiment
        experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}_nens_{n_ens}")
        
        calib_path = os.path.join(experiment_dir, "calibrated_thresholds.npy")
        if not os.path.exists(calib_path):
            print(f"Skipping: Thresholds not found at {calib_path}")
            continue
            
        Q_p = jnp.load(calib_path)
        tau_lo = alpha / 2
        tau_hi = 1 - (alpha / 2)

        # 2. Restoration Logic (Matching your Callback's Composite structure)
        def restore_trainer(label, tau, current_n_ens):
            q_dir = os.path.join(experiment_dir, label)
            da_model = QuantileEnsembleDAModel(
                model=unet,
                input_shape=(cfg.batch_size, cfg.res, cfg.res, cfg.res, out_dim),
                n_ens=current_n_ens,
                tau=tau
            )
            if cfg.optimizer=="adamw":
                tx = optax.adamw(learning_rate=cfg.learning_rate)
            elif cfg.optimizer=="asgld":
                tx = asgld(learning_rate=cfg.learning_rate)
            elif cfg.optimizer=="adamsgld":
                tx = adam_sgld(learning_rate=cfg.learning_rate)
            elif cfg.optimizer=="adamwnoise":
                tx = optax.chain(
                    optax.adamw(learning_rate=cfg.learning_rate,weight_decay=0.0),
                    add_langevin_noise(learning_rate=cfg.learning_rate,seed=123)
                    )
            trainer = BeltramiTrainer(
                model=da_model, 
                optimizer=tx, 
                rng=jax.random.PRNGKey(cfg.seed)
            )
            
            # The folder prefix in your class is "checkpoints"
            mngr = ocp.CheckpointManager(os.path.join(q_dir, "checkpoints"))
            
            # Match the Composite save structure: {'default': StandardRestore}
            # Note: your class uses "default" as the train_state_field
            trainer.train_state = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.StandardRestore(trainer.train_state)
            )
            return trainer

        trainer_lo = restore_trainer("lo", tau_lo, n_ens)
        trainer_hi = restore_trainer("hi", tau_hi, n_ens)

        if cfg.type=="beltrami":
            
            gt, obs, coords, times = generate_beltrami_da_data(
                spatial_res=cfg.res, 
                time_steps=cfg.time_steps,
                dt=cfg.dt, 
                a=1.0,
                d=1.0,
                noise_std=cfg.noise_std
            )    
        elif cfg.type=="ks3d":
            
            gt, obs, times = generate_ks3d_da_data(
                spatial_res=cfg.res, 
                time_steps=cfg.time_steps,
                noise_std=cfg.noise_std
            )
        
        start_idx = 2
        all_preds_lo, all_preds_hi, actual_truth = [], [], []

        for i in range(cfg.num_unroll_steps):
            current_state = obs[start_idx + (2*i) : start_idx + + (2*i) + 1]
            batch_size = current_state.shape[0]
            dt_val = times[2] - times[0]
            dt_input = jnp.full((batch_size,), dt_val)
            
            p_lo = get_ensemble_mean_pred(trainer_lo, current_state, dt_input)
            p_hi = get_ensemble_mean_pred(trainer_hi, current_state, dt_input)

            all_preds_lo.append(p_lo - Q_p)
            all_preds_hi.append(p_hi + Q_p)
            actual_truth.append(gt[start_idx + (2*i) + 2])

        # 4. Coverage Calculation
        preds_lo = jnp.concatenate(all_preds_lo, axis=0)
        preds_hi = jnp.concatenate(all_preds_hi, axis=0)
        truth = jnp.stack(actual_truth)
        
        # ── cube scatter plots ────────────────────────────────────────────────
        cube_dir = os.path.join(workdir_root, "analysis_plots", "cubes")
        n_time   = truth.shape[0]
 
        for t in range(n_time):
            lo_vol = [np.array(preds_lo[t, ..., f]) for f in range(out_dim)]
            hi_vol = [np.array(preds_hi[t, ..., f]) for f in range(out_dim)]
            gt_vol = [np.array(truth[t, ..., f]) for f in range(out_dim)]

        
        field_names = ['u', 'v', 'w', 'p'] if cfg.type=="beltrami" else ['u']

        export_all_videos(
            lo_vol=lo_vol,
            hi_vol=hi_vol,
            gt_vol=gt_vol,
            field_names=field_names,
            alpha=alpha,
            n_ens=n_ens,
            lr=cfg.learning_rate,
            fps=12,
            duration=3,
            out_dir=cube_dir
        )


def main(_):
    run_analysis(FLAGS.config)

if __name__ == "__main__":
    app.run(main)