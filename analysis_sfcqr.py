import os

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
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
import pyvista as pv

from model.unet3d import UNet3d
from train.models import QuantileEnsembleDAModel  # Use the Ensemble Model
from train.trainers import BeltramiTrainer
from data.beltramiDataGeneration import *
from data.ks3dDataGeneration import *
from utils.plot_utils import plot_line

from train.optimizer import *

# os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Set pyvista environment
os.environ["PYVISTA_OFF_SCREEN"] = "true"
pv.OFF_SCREEN = True

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
        # params has a leading batch dim of 1
        params = jax.tree_util.tree_map(lambda x: x[0], variables["params"])
        return trainer.model.model.apply(
            {"params": params}, obs, time=dt_input, is_training=False
        )
    
    ensemble_preds = jax.vmap(single_apply)(variables["params"])
    return jnp.mean(ensemble_preds, axis=0)

def get_ensemble_random_pred(trainer, obs, dt_input, key):
    variables = trainer.train_state.model_variables

    def single_apply(p):
        return trainer.model.model.apply(
            {"params": p}, obs, time=dt_input, is_training=False
        )

    # If only 1 ensemble member, skip vmap entirely (nothing to randomize over)
    if trainer.model.n_ens == 1:
        # params has a leading batch dim of 1
        params = jax.tree_util.tree_map(lambda x: x[0], variables["params"])
        return trainer.model.model.apply(
            {"params": params}, obs, time=dt_input, is_training=False
        )

    ensemble_preds = jax.vmap(single_apply)(variables["params"])
    # Randomly select one ensemble member's prediction instead of averaging
    # rand_idx = jax.random.randint(key, shape=(), minval=0, maxval=trainer.model.n_ens)
    return ensemble_preds

FIELD_NAMES = ["u", "v", "w", "p"]
CMAP        = "RdBu_r"
CMAP        = "coolwarm"
DPI         = 100
MARKER_SIZE = 14   # scatter dot size — increase for coarser grids


# RENDER_W     = 16      # vape4d render resolution (pixels)
# RENDER_H     = 16
# DPI          = 90
# VIDEO_FPS    = 4        # frames per second in output MP4


# ── PyVista 3-D volume cube ───────────────────────────────────────────────────

def _add_pyvista_volume(plotter, row, col, vol, clim, cmap, title):
    """
    Render vol (res, res, res) as a cleaner PyVista volume in the given subplot.
    """
    vol = np.asarray(vol)

    volume = pv.wrap(vol)

    plotter.subplot(row, col)

    # Cleaner volume appearance:
    # - shade=False avoids grey muddy lighting
    # - custom opacity makes near-zero values more transparent
    # - stronger absolute values become more visible
    plotter.add_volume(
        volume,
        cmap=cmap,
        opacity=[0.70, 0.35, 0.04, 0.35, 0.70],
        clim=clim,
        shade=False,
        show_scalar_bar=False,
    )

    # Add a subtle box so the cube boundary is clearer
    plotter.add_bounding_box(
        color="black",
        line_width=1,
        opacity=0.25,
    )

    # Put title near top-left of each subplot, small and non-overlapping
    txt = plotter.add_text(
        title,
        position=(0.04, 0.92),
        viewport=True,
        font_size=8,
        color="#222222",
    )

    # Make text less harsh
    try:
        txt.GetTextProperty().SetBold(False)
        txt.GetTextProperty().SetShadow(False)
    except Exception:
        pass

    plotter.camera_position = "iso"

    # Zoom OUT slightly to leave room for the title
    plotter.camera.zoom(0.82)
    
# ── per-timestep PyVista volume figure ────────────────────────────────────────

def plot_interval_cubes(t, n_fields, lo_vol, hi_vol, gt_vol, folder, alpha, n_ens, lr):
    """
    lo_vol, hi_vol, gt_vol : list of arrays, each (res, res, res).

    Layout:
        Row 0: lower bound
        Row 1: upper bound
        Row 2: ground truth

        Columns:
            u, v, w, p for Beltrami
            u only for KS3D
    """

    row_labels = ["lower bound", "upper bound", "ground truth"]
    row_names_for_title = ["lower", "upper", "truth"]

    # Global symmetric colour scale across all rows and fields.
    # This keeps the colours comparable between lower, upper and ground truth.
    all_vals = np.concatenate(
        [np.asarray(v).ravel() for v in lo_vol + hi_vol + gt_vol]
    )

    vmax = float(np.max(np.abs(all_vals)))

    # Avoid zero-width colour range if the volume is constant/zero
    if vmax == 0.0:
        vmax = 1.0

    clim = [-vmax, vmax]

    os.makedirs(folder, exist_ok=True)
    lr_str = str(lr).replace(".", "p")
    fname = os.path.join(
        folder, f"cubes_alpha{alpha}_nens{n_ens}_lr{lr_str}_t{t:03d}.png"
    )

    # Window size roughly replaces the old Matplotlib figsize.
    # Increase these numbers if you want higher-resolution images.
    window_width = max(1200, 360 * n_fields)
    window_height = 900

    
    plotter = pv.Plotter(
        off_screen=True,
        shape=(3, n_fields),
        window_size=(window_width, window_height),
        border=False,
    )

    # Optional: white background, matching your old Matplotlib figure
    plotter.set_background("white")

    for f in range(n_fields):
        vols = [lo_vol[f], hi_vol[f], gt_vol[f]]

        for row in range(3):
            field_name = FIELD_NAMES[f] if f < len(FIELD_NAMES) else f"field {f}"

            if f == 0:
                title = f"{row_labels[row]}  ·  {field_name} {row_names_for_title[row]}"
            else:
                title = f"{field_name} {row_names_for_title[row]}"

            _add_pyvista_volume(
                plotter=plotter,
                row=row,
                col=f,
                vol=vols[row],
                clim=clim,
                cmap=CMAP,
                title=title,
            )

    # Add one scalar bar for the whole plot.
    # Since every volume uses the same clim and cmap, this scalar bar applies globally.
    plotter.add_scalar_bar(
        title="field value",
        vertical=True,
        position_x=0.91,
        position_y=0.18,
        width=0.04,
        height=0.65,
        title_font_size=12,
        label_font_size=10,
        color="black",
    )

    # Add global title.
    plotter.add_text(
        f"Conformal prediction intervals  ·  α={alpha}  s={n_ens}  t={t}",
        position="upper_left",
        font_size=12,
        color="black",
    )

    
    # plotter.render()
    # plotter.screenshot(fname)
    # plotter.close()
    
    
    try:
        plotter.show(screenshot=fname, auto_close=True)
    finally:
        plotter.close()



    print(f"  saved {fname}")


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

    # Dictionary to store results for final comparison
    coverage_list = []
    s_hmap_list = []
    alpha_hmap_list = []
    interval_size_list = []

    for alpha, n_ens in product(cfg.alpha_list, cfg.n_ens_list):
        print(f"\n>>> Analyzing Alpha: {alpha} | N_ens: {n_ens}")
        
        # 1. Path Setup for this specific experiment
        experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}_nens_{n_ens}")
        
        calib_path = os.path.join(experiment_dir, "calibrated_thresholds.npy")
        if not os.path.exists(calib_path):
            print(f"Skipping: Thresholds not found at {calib_path}")
            continue
            
        Q_p = jnp.load(calib_path)
        print(Q_p)
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
            
            
            if cfg.ens_type=="mean":
                p_lo = get_ensemble_mean_pred(trainer_lo, current_state, dt_input)
                p_hi = get_ensemble_mean_pred(trainer_hi, current_state, dt_input)
            elif cfg.ens_type=="random":
                key = jax.random.PRNGKey(123)
                p_lo_rnd = get_ensemble_random_pred(trainer_lo, current_state, dt_input, key)
                p_hi_rnd = get_ensemble_random_pred(trainer_hi, current_state, dt_input, key)

                # Check this once if the rand_idx is being chosen properly
                p_lo_sorted = jnp.minimum(p_lo_rnd, p_hi_rnd)
                p_hi_sorted = jnp.maximum(p_lo_rnd, p_hi_rnd)
                rand_idx = jax.random.randint(key, shape=(), minval=0, maxval=trainer_lo.model.n_ens)
                if trainer_lo.model.n_ens == 1:
                    p_lo = p_lo_sorted[rand_idx,:,:,:,:][None,...]
                    p_hi = p_hi_sorted[rand_idx,:,:,:,:][None,...]
                else:
                    p_lo = p_lo_sorted[rand_idx,:,:,:,:]
                    p_hi = p_hi_sorted[rand_idx,:,:,:,:]
            

            all_preds_lo.append(p_lo - Q_p)
            all_preds_hi.append(p_hi + Q_p)
            actual_truth.append(gt[start_idx + (2*i) + 2])

        # 4. Coverage Calculation
        preds_lo = jnp.concatenate(all_preds_lo, axis=0)
        preds_hi = jnp.concatenate(all_preds_hi, axis=0)
        truth = jnp.stack(actual_truth)
        
        print(preds_lo.shape)
        
        # # ── cube scatter plots ────────────────────────────────────────────────
        # cube_dir = os.path.join(workdir_root, "analysis_plots", "cubes")
        # n_time   = truth.shape[0]
 
        # for t in range(n_time):
        #     lo_vol = [np.array(preds_lo[t, ..., f]) for f in range(out_dim)]
        #     hi_vol = [np.array(preds_hi[t, ..., f]) for f in range(out_dim)]
        #     gt_vol = [np.array(truth[t, ..., f]) for f in range(out_dim)]
 
        #     plot_interval_cubes(
        #         t=(2*t)+2,
        #         n_fields=out_dim,
        #         lo_vol=lo_vol,
        #         hi_vol=hi_vol,
        #         gt_vol=gt_vol,
        #         folder=cube_dir,
        #         alpha=alpha,
        #         n_ens=n_ens,
        #         lr=cfg.learning_rate,
        #     )
 
        # ── scalar metrics ────────────────────────────────────────────────────
        is_covered         = (truth >= preds_lo) & (truth <= preds_hi)
        interval_widths    = preds_hi - preds_lo
 
        is_covered = (truth >= preds_lo) & (truth <= preds_hi)
        if cfg.type=="beltrami":
            coverage = jnp.mean(is_covered, axis=(0, 1, 2, 3))
            avg_interval_sizes = jnp.mean(interval_widths, axis=(0, 1, 2, 3))
        elif cfg.type=="ks3d":
            coverage = jnp.mean(is_covered, axis=(0))
            avg_interval_sizes = jnp.mean(interval_widths, axis=(0))
        coverage_list.append(coverage)
        s_hmap_list.append(n_ens)
        alpha_hmap_list.append(alpha)
        interval_size_list.append(avg_interval_sizes)
        
    coverages = np.array(coverage_list)
    intervals = np.array(interval_size_list)
    s_vals = np.array(s_hmap_list)
    alpha_vals = np.array(alpha_hmap_list)
    target_coverage = 1.0 - alpha_vals

    channel_names = ['u', 'v', 'w', 'p'] if cfg.type=="beltrami" else ['u']
    lr_suffix = f"lr{cfg.learning_rate}"
    folder_name = os.path.join(workdir_root, "analysis_plots")
    os.makedirs(folder_name, exist_ok=True)

    offset = 0

    # PLOTTING PER CHANNEL
    for i, name in enumerate(channel_names):
        chan_coverage = coverages[:, i]
        chan_intervals = intervals[:, i]
        
        print(f"\nFinal Metrics for Channel {name}:")
        print(f"Mean Coverage: {np.mean(chan_coverage)*100:.2f}%")

        # Plot 1: Coverage vs Ensemble Size (s)
        # We group by alpha so each alpha gets a separate line
        filename = f"SCQR_Coverage_lineplot_{name}_{lr_suffix}"
        plot_line(
            x_list=s_vals, 
            x_label=r'$s$', 
            y_list=chan_coverage, 
            y_label=f"Coverage Fraction ({name})", 
            line_list=target_coverage, 
            line_label=r'$1 - \alpha$', 
            folder_name=folder_name, 
            filename=filename, 
            draw_line=True, 
            offset=offset
        )

        # Plot 2: Interval Size vs Ensemble Size (s)
        filename = f"SCQR_Interval_lineplot_{name}_{lr_suffix}"
        plot_line(
            x_list=s_vals, 
            x_label=r'$s$', 
            y_list=chan_intervals, 
            y_label=f"Interval Size ({name})", 
            line_list=target_coverage, 
            line_label=r'$1 - \alpha$', 
            folder_name=folder_name, 
            filename=filename
        )

        # Plot 3: Interval Size vs Coverage
        filename = f"SCQR_Interval_Coverage_lineplot_{name}_{lr_suffix}"
        plot_line(
            x_list=chan_coverage, 
            x_label='Coverage Fraction', 
            y_list=chan_intervals, 
            y_label="Interval Size", 
            line_list=s_vals, 
            line_label=r'$s$', 
            folder_name=folder_name, 
            filename=filename, 
            shaded=True, 
            draw_c_true=True
        )



def main(_):
    run_analysis(FLAGS.config)

if __name__ == "__main__":
    app.run(main)