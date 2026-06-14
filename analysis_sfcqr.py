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
# from data.beltramiDataGeneration import generate_da_data
from data.ks3dDataGeneration import generate_da_data
from utils.plot_utils import plot_line

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


# ── 3-D scatter cube ──────────────────────────────────────────────────────────
 
def _draw_scatter_cube(ax, vol, norm, cmap):
    """
    Render vol (res, res, res) as a 3-D scatter plot.
    Each grid point is a dot coloured by its scalar value.
    """
    res = vol.shape[0]
    xi, yi, zi = np.meshgrid(
        np.arange(res), np.arange(res), np.arange(res), indexing="ij"
    )
    colors = mpl.cm.get_cmap(cmap)(norm(vol.ravel()))
 
    ax.scatter(
        xi.ravel(), yi.ravel(), zi.ravel(),
        c=colors,
        s=MARKER_SIZE,
        marker="o",
        linewidths=0,
        depthshade=True,
    )
 
    ax.set_xlim(0, res - 1)
    ax.set_ylim(0, res - 1)
    ax.set_zlim(0, res - 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.view_init(elev=22, azim=35)
 
 
# ── per-timestep figure ───────────────────────────────────────────────────────
 
def plot_interval_cubes(t, lo_vol, hi_vol, gt_vol, folder, alpha, n_ens, lr):
    """
    lo_vol, hi_vol, gt_vol : list of 4 arrays, each (res, res, res).
 
    Layout  (3 rows x 4 fields + ONE shared colorbar on the far right)
    ───────────────────────────────────────────────────────────────────
    Row 0  lower bound  │ u_lo  v_lo  w_lo  p_lo │
    Row 1  upper bound  │ u_hi  v_hi  w_hi  p_hi │  ← single colorbar
    Row 2  ground truth │ u_gt  v_gt  w_gt  p_gt │
    """
    n_fields   = 1
    row_labels = ["lower bound", "upper bound", "ground truth"]
    row_colors = ["#3b82f6", "#ef4444", "#16a34a"]
 
    # One global symmetric colour scale across ALL fields and rows
    all_vals = np.concatenate(
        [v.ravel() for v in lo_vol + hi_vol + gt_vol]
    )
    vmax = float(np.max(np.abs(all_vals)))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = mpl.colormaps[CMAP]
 
    # GridSpec: 4 cube columns + 1 thin colorbar column
    width_ratios = [6] * n_fields + [0.5]
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs  = gridspec.GridSpec(3, n_fields + 1,
                            width_ratios=width_ratios,
                            wspace=0.02, hspace=0.05,
                            left=0.06, right=0.97,
                            top=0.92, bottom=0.02)
 
    for f in range(n_fields):
        vols      = [lo_vol[f], hi_vol[f], gt_vol[f]]
        subtitles = [
            f"{FIELD_NAMES[f]}  lower",
            f"{FIELD_NAMES[f]}  upper",
            f"{FIELD_NAMES[f]}  truth",
        ]
        for row in range(3):
            ax = fig.add_subplot(gs[row, f], projection="3d")
            _draw_scatter_cube(ax, vols[row], norm, cmap)
            ax.set_title(subtitles[row],
                         fontsize=8, pad=1,
                         color=row_colors[row], fontweight="500")
 
    # single shared colorbar on the far right, spanning all rows
    cax = fig.add_subplot(gs[:, n_fields])
    cb  = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                       cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("field value", fontsize=8, labelpad=4)
 
    # rotated row labels on left margin
    for row, (label, color) in enumerate(zip(row_labels, row_colors)):
        y_pos = 0.87 - row * 0.295
        fig.text(0.005, y_pos, label,
                 va="center", ha="left", fontsize=9,
                 fontweight="bold", color=color, rotation=90)
 
    fig.suptitle(
        f"Beltrami conformal prediction intervals  ·  "
        f"α={alpha}  s={n_ens}  t={t}",
        fontsize=10, fontweight="500", y=0.97
    )
 
    os.makedirs(folder, exist_ok=True)
    lr_str = str(lr).replace(".", "p")
    fname  = os.path.join(
        folder, f"cubes_alpha{alpha}_nens{n_ens}_lr{lr_str}_t{t:03d}.png"
    )
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")

# # ── vape4d volume render ──────────────────────────────────────────────────────
 
# def _render_volume(vol, vmin, vmax, cmap_base="RdBu_r"):
#     """
#     Render a single (res, res, res) volume with vape4d.
#     Values are normalised to [0,1] using the global vmin/vmax so that
#     the colour scale is identical across all panels and timesteps.
#     Returns (H, W, 4) uint8 RGBA.
#     """
#     vol_norm = (vol.astype(np.float32) - vmin) / (vmax - vmin + 1e-8)
#     vol_norm = np.clip(vol_norm, 0.0, 1.0)
#     data     = vol_norm[np.newaxis]   # (1, res, res, res)  [T,D,H,W]
#     colormap = diverging_alpha(mpl.colormaps[cmap_base])
#     return render(data, colormap, width=RENDER_W, height=RENDER_H)
 
 
# # ── per-timestep figure ───────────────────────────────────────────────────────

# def plot_interval_cubes(t, lo_vol, hi_vol, gt_vol,
#                         vmin_global, vmax_global, norm,
#                         folder, alpha, n_ens, lr):
#     """
#     Render and save one PNG for timestep t.
 
#     lo_vol, hi_vol, gt_vol : list of 4 arrays, each (res, res, res).
#     vmin_global, vmax_global, norm : shared colour scale (pre-computed outside).
 
#     Layout  (3 rows x 4 fields + 1 shared colorbar)
#     ────────────────────────────────────────────────
#     Row 0  lower bound  │ u_lo  v_lo  w_lo  p_lo │
#     Row 1  upper bound  │ u_hi  v_hi  w_hi  p_hi │  ← single colorbar
#     Row 2  ground truth │ u_gt  v_gt  w_gt  p_gt │
#     """
#     row_labels = ["lower bound", "upper bound", "ground truth"]
#     row_colors = ["#3b82f6", "#ef4444", "#16a34a"]
#     all_rows   = [lo_vol, hi_vol, gt_vol]
 
#     # render all 12 volumes
#     rendered = [
#         [_render_volume(vol, vmin_global, vmax_global) for vol in row_vols]
#         for row_vols in all_rows
#     ]
 
#     width_ratios = [6] * 4 + [0.45]
#     fig = plt.figure(figsize=(17, 9), facecolor="white")
#     gs  = gridspec.GridSpec(3, 5,
#                             width_ratios=width_ratios,
#                             wspace=0.02, hspace=0.06,
#                             left=0.06, right=0.97,
#                             top=0.92, bottom=0.02)
 
#     for row in range(3):
#         for f in range(4):
#             ax = fig.add_subplot(gs[row, f])
#             ax.imshow(rendered[row][f])
#             ax.set_axis_off()
#             ax.set_title(
#                 f"{FIELD_NAMES[f]}  {['lower','upper','truth'][row]}",
#                 fontsize=8, pad=2,
#                 color=row_colors[row], fontweight="500"
#             )
 
#     # single shared colorbar
#     cax = fig.add_subplot(gs[:, 4])
#     cb  = fig.colorbar(ScalarMappable(norm=norm, cmap=mpl.colormaps["RdBu_r"]),
#                        cax=cax, orientation="vertical")
#     cb.ax.tick_params(labelsize=7)
#     cb.set_label("field value", fontsize=8, labelpad=4)
 
#     for row, (label, color) in enumerate(zip(row_labels, row_colors)):
#         fig.text(0.005, 0.87 - row * 0.295, label,
#                  va="center", ha="left", fontsize=9,
#                  fontweight="bold", color=color, rotation=90)
 
#     fig.suptitle(
#         f"Beltrami conformal prediction intervals  ·  "
#         f"α={alpha}  s={n_ens}  t={t}",
#         fontsize=10, fontweight="500", y=0.97
#     )
 
#     os.makedirs(folder, exist_ok=True)
#     lr_str = str(lr).replace(".", "p")
#     fname  = os.path.join(
#         folder, f"cubes_alpha{alpha}_nens{n_ens}_lr{lr_str}_t{t:03d}.png"
#     )
#     fig.savefig(fname, dpi=DPI, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  saved {fname}")
#     return fname

 
# ── video ─────────────────────────────────────────────────────────────────────
 
# def _natural_sort_key(s):
#     return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]
 
 
# def make_video(png_paths, output_path, fps=VIDEO_FPS):
#     """Stitch a list of PNG paths (in order) into an MP4 via FFMpegWriter."""
#     if not png_paths:
#         print("  No frames to encode — skipping video.")
#         return
 
#     img0  = np.array(Image.open(png_paths[0]))
#     h, w  = img0.shape[:2]
#     fig, ax = plt.subplots(figsize=(w / 100, h / 100), facecolor="white")
#     ax.axis("off")
#     fig.subplots_adjust(0, 0, 1, 1)
 
#     im  = ax.imshow(img0)
#     txt = ax.text(0.01, 0.99, "", transform=ax.transAxes,
#                   fontsize=9, va="top", ha="left", color="white",
#                   bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))
 
#     def update(i):
#         img = np.array(Image.open(png_paths[i]))
#         im.set_data(img)
#         txt.set_text(f"t = {i:03d} / {len(png_paths)-1:03d}")
#         return im, txt
 
#     ani = animation.FuncAnimation(
#         fig, update,
#         frames=len(png_paths),
#         interval=1000 / fps,
#         blit=True
#     )
 
#     os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
 
#     writer = animation.FFMpegWriter(
#         fps=fps,
#         codec="libx264",
#         extra_args=["-pix_fmt", "yuv420p", "-crf", "18"]
#     )
#     print(f"  encoding {len(png_paths)} frames → {output_path}")
#     ani.save(output_path, writer=writer, dpi=100)
#     plt.close(fig)
#     print(f"  video saved.")

def run_analysis(cfg):
    workdir_root = os.path.abspath(cfg.workdir)
    
    # Define UNet outside the loops for efficiency
    unet = UNet3d(
        out_channels=1,
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
        tau_lo = alpha / 2
        tau_hi = 1 - (alpha / 2)

        # 2. Restoration Logic (Matching your Callback's Composite structure)
        def restore_trainer(label, tau, current_n_ens):
            q_dir = os.path.join(experiment_dir, label)
            da_model = QuantileEnsembleDAModel(
                model=unet,
                input_shape=(cfg.batch_size, cfg.res, cfg.res, cfg.res, 1),
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

        # 3. Data Generation & Unrolling
        gt, obs, times = generate_da_data(
            spatial_res=cfg.res, 
            time_steps=100, 
            # dt=cfg.dt,
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
            actual_truth.append(gt[start_idx + + (2*i) + 2])

        # 4. Coverage Calculation
        preds_lo = jnp.concatenate(all_preds_lo, axis=0)
        preds_hi = jnp.concatenate(all_preds_hi, axis=0)
        truth = jnp.stack(actual_truth)
        
        # ── cube scatter plots ────────────────────────────────────────────────
        cube_dir = os.path.join(workdir_root, "analysis_plots", "cubes")
        n_time   = truth.shape[0]
 
        for t in range(n_time):
            lo_vol = [np.array(preds_lo[t, ..., f]) for f in range(1)]
            hi_vol = [np.array(preds_hi[t, ..., f]) for f in range(1)]
            gt_vol = [np.array(truth[t, ..., f]) for f in range(1)]
 
            plot_interval_cubes(
                t=(2*t)+2,
                lo_vol=lo_vol,
                hi_vol=hi_vol,
                gt_vol=gt_vol,
                folder=cube_dir,
                alpha=alpha,
                n_ens=n_ens,
                lr=cfg.learning_rate,
            )
        
        # ── compute ONE global colour scale for this (alpha, n_ens) run ───────
        # all_vals    = np.concatenate([
        #     np.array(preds_lo).ravel(),
        #     np.array(preds_hi).ravel(),
        #     np.array(truth).ravel(),
        # ])
        # vmax_global = float(np.max(np.abs(all_vals)))
        # vmin_global = -vmax_global
        # norm        = Normalize(vmin=vmin_global, vmax=vmax_global)
 
        # ── cube plots + collect PNG paths for video ──────────────────────────
        # cube_dir  = os.path.join(workdir_root, "analysis_plots", "cubes")
        # n_time    = truth.shape[0]
        # png_paths = []
 
        # for t in range(n_time):
        #     lo_vol = [np.array(preds_lo[t, ..., f]) for f in range(4)]
        #     hi_vol = [np.array(preds_hi[t, ..., f]) for f in range(4)]
        #     gt_vol = [np.array(truth[t,    ..., f]) for f in range(4)]
 
        #     fname = plot_interval_cubes(
        #         t=t,
        #         lo_vol=lo_vol,
        #         hi_vol=hi_vol,
        #         gt_vol=gt_vol,
        #         vmin_global=vmin_global,
        #         vmax_global=vmax_global,
        #         norm=norm,
        #         folder=cube_dir,
        #         alpha=alpha,
        #         n_ens=n_ens,
        #         lr=cfg.learning_rate,
        #     )
        #     png_paths.append(fname)
 
        # # ── stitch PNGs into MP4 ──────────────────────────────────────────────
        # lr_str    = str(cfg.learning_rate).replace(".", "p")
        # video_dir = os.path.join(workdir_root, "analysis_plots", "videos")
        # video_out = os.path.join(
        #     video_dir,
        #     f"beltrami_alpha{alpha}_nens{n_ens}_lr{lr_str}.mp4"
        # )
        # make_video(png_paths, video_out, fps=VIDEO_FPS)
 
        # ── scalar metrics ────────────────────────────────────────────────────
        is_covered         = (truth >= preds_lo) & (truth <= preds_hi)
        coverage           = jnp.mean(is_covered, axis=(0, 1, 2, 3))
        interval_widths    = preds_hi - preds_lo
        avg_interval_sizes = jnp.mean(interval_widths, axis=(0, 1, 2, 3))
 
        coverage_list.append(coverage)
        s_hmap_list.append(n_ens)
        alpha_hmap_list.append(alpha)
        interval_size_list.append(avg_interval_sizes)

        is_covered = (truth >= preds_lo) & (truth <= preds_hi)
        coverage = jnp.mean(is_covered, axis=(0, 1, 2, 3))
        
        interval_widths = preds_hi - preds_lo
        
        avg_interval_sizes = jnp.mean(interval_widths, axis=(0, 1, 2, 3))
        
    
        coverage_list.append(coverage)
        s_hmap_list.append(n_ens)
        alpha_hmap_list.append(alpha)
        interval_size_list.append(avg_interval_sizes)
        
    coverages = np.array(coverage_list)
    intervals = np.array(interval_size_list)
    s_vals = np.array(s_hmap_list)
    alpha_vals = np.array(alpha_hmap_list)
    target_coverage = 1.0 - alpha_vals

    # channel_names = ['u', 'v', 'w', 'p']
    channel_names = ['u']
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