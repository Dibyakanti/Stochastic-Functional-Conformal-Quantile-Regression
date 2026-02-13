import os
import sys

# --- 1. Environment Setup ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp

jax.config.update("jax_enable_x64", False)
sys.path.append("./")

from model.unet3d import UNet3d
from train.models import DataAssimilationModel
from train.trainers import BeltramiTrainer
from data.beltramiDataGeneration import generate_da_data, beltrami_da_dataloader

# --- 2. Advanced Visualization Logic ---
def _plot_comparison_grid(preds, truth, times, start_idx, plot_dir):
    """Plots Truth vs Pred every 20 steps and a separate error plot."""
    os.makedirs(plot_dir, exist_ok=True)
    step_indices = np.arange(0, len(preds), 20)
    num_cols = len(step_indices)
    mid_z = preds.shape[2] // 2
    
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
    
    for i, step in enumerate(step_indices):
        t_val = times[start_idx + step]
        im_gt = axes[0, i].imshow(truth[step, :, :, mid_z, 0], cmap='magma')
        axes[0, i].set_title(f"Truth t={t_val:.2f}")
        plt.colorbar(im_gt, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        im_pred = axes[1, i].imshow(preds[step, :, :, mid_z, 0], cmap='magma')
        axes[1, i].set_title(f"Pred t={t_val:.2f}")
        plt.colorbar(im_pred, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        for ax in axes[:, i]:
            ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("GROUND TRUTH", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("PREDICTED", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comparison_grid.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    diff_norm = jnp.sqrt(jnp.sum(jnp.square(preds - truth), axis=(1, 2, 3, 4)))
    truth_norm = jnp.sqrt(jnp.sum(jnp.square(truth), axis=(1, 2, 3, 4)))
    rel_l2_error = diff_norm / truth_norm
    
    plt.plot(range(len(rel_l2_error)), rel_l2_error, color='crimson', lw=2, label='Relative L2 Error')
    plt.yscale('log')
    plt.xlabel("Autoregressive Steps")
    plt.ylabel("Relative L2 Error")
    plt.title("Prediction Error Drift Over Time")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "error_plot.png"), dpi=300)
    plt.close()

def _plot_comparison_grid_3d(preds, truth, times, start_idx, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    step_indices = np.arange(0, len(preds), 20)
    num_cols = len(step_indices)
    mag_truth = jnp.sqrt(jnp.sum(jnp.square(truth[..., :3]), axis=-1))
    mag_preds = jnp.sqrt(jnp.sum(jnp.square(preds[..., :3]), axis=-1))
    
    fig = plt.figure(figsize=(5 * num_cols, 10))
    for i, step in enumerate(step_indices):
        t_val = times[start_idx + step]
        ax_gt = fig.add_subplot(2, num_cols, i + 1, projection='3d')
        threshold = mag_truth[step].mean() * 1.2
        vol_gt = mag_truth[step] > threshold
        ax_gt.voxels(vol_gt, facecolors=plt.cm.magma(mag_truth[step]/mag_truth[step].max()), 
                     edgecolor='k', linewidth=0.05, alpha=0.3)
        ax_gt.set_title(f"Truth t={t_val:.2f}")
        
        ax_pred = fig.add_subplot(2, num_cols, i + 1 + num_cols, projection='3d')
        vol_pred = mag_preds[step] > threshold
        ax_pred.voxels(vol_pred, facecolors=plt.cm.magma(mag_preds[step]/mag_preds[step].max()), 
                       edgecolor='k', linewidth=0.05, alpha=0.3)
        ax_pred.set_title(f"Pred t={t_val:.2f}")
        for ax in [ax_gt, ax_pred]:
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)

    plt.suptitle("3D Volumetric Evolution (Velocity Magnitude)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comparison_3d_volumetric.png"), dpi=200)
    plt.close()

def _vol_plot_comparison_grid_3d(preds, truth, times, start_idx, plot_dir, component_idx=0):
    os.makedirs(plot_dir, exist_ok=True)
    
    comp_names = {0: 'u-velocity', 1: 'v-velocity', 2: 'w-velocity', 3: 'pressure'}
    name = comp_names.get(component_idx, f'channel_{component_idx}')
    
    step_indices = np.arange(0, len(preds), 20)
    num_cols = len(step_indices)
    
    data_truth = truth[..., component_idx]
    data_preds = preds[..., component_idx]
    
    # 1. Calculate global min/max for a consistent scale
    vmin = min(data_truth.min(), data_preds.min())
    vmax = max(data_truth.max(), data_preds.max())
    
    fig = plt.figure(figsize=(6 * num_cols, 13), facecolor='white')
    res = data_truth.shape[1]
    x, y, z = np.mgrid[0:res, 0:res, 0:res]

    img = None # Variable to store the mappable for the colorbar
    for i, step in enumerate(step_indices):
        t_val = times[start_idx + step]
        for row, (data_slice, label) in enumerate([(data_truth, "Truth"), (data_preds, "Pred")]):
            ax = fig.add_subplot(2, num_cols, i + 1 + (row * num_cols), projection='3d')
            
            val = data_slice[step]
            mask = np.abs(val) > 0 # Adjust threshold as needed
            
            # 2. Pass vmin and vmax to keep scales identical
            img = ax.scatter(x[mask], y[mask], z[mask], 
                             c=val[mask], 
                             cmap='turbo', 
                             vmin=vmin, 
                             vmax=vmax,
                             s=6, 
                             alpha=0.4, 
                             edgecolors='none')
            
            ax.set_title(f"{label} (t={t_val:.2f})")

    # 3. Add the shared colorbar
    # pad: distance from plots; shrink: size of bar; aspect: thickness
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02]) # [left, bottom, width, height]
    fig.colorbar(img, cax=cbar_ax, orientation='horizontal', label=f'{name} Magnitude')
    
    plt.savefig(os.path.join(plot_dir, f"{name}_comparison.png"), bbox_inches='tight')
    plt.close()
    
# --- 3. Main Analysis Execution ---
def run_analysis(checkpoint_dir: str, spatial_res: int = 64): # <--- INCREASED DEFAULT HERE
    ckpt_path = os.path.abspath(checkpoint_dir)
    experiment_dir = os.path.dirname(ckpt_path.rstrip('/')) 
    
    # UNet architecture can handle arbitrary spatial_res if fully convolutional
    unet_config = UNet3d(
        out_channels=4,
        dtype=jnp.float32,
        time_cond=True,
        num_channels=(32, 64, 128, 256),
        use_spatial_attention=(False, False, True, True)
    )
    
    # Update input shape to match high resolution
    da_model = DataAssimilationModel(model=unet_config, input_shape=(1, spatial_res, spatial_res, spatial_res, 4))
    trainer = BeltramiTrainer(model=da_model, optimizer=optax.adamw(1e-4), rng=jax.random.PRNGKey(0))

    mngr = ocp.CheckpointManager(ckpt_path)
    latest_step = mngr.latest_step()
    if latest_step is None: return print(f"No checkpoints in {ckpt_path}")
    
    trainer.train_state = mngr.restore(latest_step, args=ocp.args.StandardRestore(trainer.train_state))
    print(f"Restored from step {latest_step}. Analyzing at {spatial_res}^3 resolution.")

    # Generate data at the new higher spatial resolution
    gt, obs, coords, times = generate_da_data(spatial_res=spatial_res, time_steps=100, noise_std=0)
    start_idx = 2 
    current_state = gt[start_idx : start_idx + 1]
    
    predictions = []
    actual_truth = [] 
    
    dt = times[1] - times[0] 
    dt_input = jnp.array([dt])

    num_unroll_steps = 80
    for i in range(num_unroll_steps):
        # The model processes high-res data autoregressively
        pred = trainer.model.model.apply(
            trainer.train_state.model_variables,
            current_state,
            time=dt_input,
            is_training=False
        )
        predictions.append(pred)
        actual_truth.append(gt[start_idx + i + 1])
        current_state = pred
        
    predictions = jnp.concatenate(predictions, axis=0)
    actual_truth = jnp.stack(actual_truth)

    plot_path = os.path.join(experiment_dir, "plots")
    
    _plot_comparison_grid(predictions, actual_truth, times, start_idx, plot_path)
    
    # _plot_comparison_grid_3d(predictions, actual_truth, times, start_idx, plot_path)
    
    _vol_plot_comparison_grid_3d(predictions, actual_truth, times, start_idx, plot_path, component_idx=0)

if __name__ == "__main__":
    # Specify the desired high resolution here (e.g., 64 or 128)
    run_analysis("./beltrami_experiment_zero_noise/checkpoints/", spatial_res=64)