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
from data.ks3dDataGeneration import generate_da_data
# from data.beltramiDataGeneration import generate_da_data
from utils.plot_utils import plot_line


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

def get_predictions(trainer, observations, times):
    """Manual prediction extractor that handles ensemble dimensions."""
    # 1. Prepare inputs
    # Use the actual batch size of the observations provided
    batch_size = observations.shape[0]
    dt = jnp.full((batch_size,), times[2] - times[0])
    
    # 2. Extract variables (params and potentially mutables/batch_stats)
    variables = trainer.train_state.model_variables
    params = variables["params"]
    # Handle mutables if they exist (e.g. Batch Norm state), otherwise empty dict
    mutables = {k: v for k, v in variables.items() if k != "params"}

    # 3. Define a single-pass function for one ensemble member
    def single_model_apply(p, m):
        return trainer.model.model.apply(
            {"params": p, **m},
            observations,
            time=dt,
            is_training=False
        )

    # 4. Use jax.vmap to iterate over the ensemble dimension
    # This maps over the leading axis of params and mutables
    ensemble_preds = jax.vmap(single_model_apply)(params, mutables)
    
    # Shape: [n_ens, N_cal, res, res, res, channels]
    return ensemble_preds

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
    
    # --- Data Generation ---
    gt, obs, times = generate_da_data(
        spatial_res=cfg.res, 
        time_steps=cfg.time_steps,
        # dt=cfg.dt, 
        # a=1.0,
        # d=1.0,
        noise_std=cfg.noise_std
    )
    
    # Split into Train and Calibration
    num_samples = gt.shape[0]
    idx = jax.random.permutation(jax.random.PRNGKey(cfg.seed), num_samples)
    train_idx, cal_idx = idx[:num_samples//2], idx[num_samples//2:]
    
    # for alpha in cfg.alpha_list:
    for alpha, n_ens in product(cfg.alpha_list, cfg.n_ens_list):
        print(f"\n>>> Analyzing Alpha: {alpha} | N_ens: {n_ens}")
        
        # 1. Path Setup for this specific experiment
        experiment_dir = os.path.join(workdir_root, f"alpha_{alpha}_nens_16")
        
        tau_lo = alpha / 2
        tau_hi = 1 - (alpha / 2)

        # 2. Restoration Logic
        # def restore_trainer(label, tau, current_n_ens):
        #     q_dir = os.path.join(experiment_dir, label)
        #     da_model = QuantileEnsembleDAModel(
        #         model=unet,
        #         input_shape=(cfg.batch_size, cfg.res, cfg.res, cfg.res, 4),
        #         n_ens=current_n_ens,
        #         tau=tau
        #     )
        #     trainer = BeltramiTrainer(
        #         model=da_model, 
        #         optimizer=optax.adamw(cfg.learning_rate), 
        #         rng=jax.random.PRNGKey(cfg.seed)
        #     )
            
        #     # The folder prefix in your class is "checkpoints"
        #     mngr = ocp.CheckpointManager(os.path.join(q_dir, "checkpoints"))
            
        #     # Match the Composite save structure: {'default': StandardRestore}
        #     # Note: your class uses "default" as the train_state_field
        #     trainer.train_state = mngr.restore(
        #         mngr.latest_step(),
        #         args=ocp.args.StandardRestore(trainer.train_state)
        #     )
        #     return trainer
        
        def restore_trainer(label, tau, current_n_ens, target_n_ens=None):
            q_dir = os.path.join(experiment_dir, label)
            
            # 1. Initialize with the ORIGINAL n_ens (so the shapes match the file on disk)
            da_model = QuantileEnsembleDAModel(
                model=unet,
                input_shape=(cfg.batch_size, cfg.res, cfg.res, cfg.res, 1),
                n_ens=current_n_ens,
                tau=tau
            )
            trainer = BeltramiTrainer(
                model=da_model, 
                optimizer=optax.adamw(cfg.learning_rate), 
                rng=jax.random.PRNGKey(cfg.seed)
            )
            
            mngr = ocp.CheckpointManager(os.path.join(q_dir, "checkpoints"))
            
            # 2. Restore the FULL ensemble
            full_state = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.StandardRestore(trainer.train_state)
            )

            # 3. Slice the state if you want fewer models (e.g., first 5)
            if target_n_ens is not None and target_n_ens < current_n_ens:
                # We slice the 'params' and 'mutables' along axis 0
                sliced_params = jax.tree_util.tree_map(lambda x: x[:target_n_ens, ...], full_state.params)
                
                # Replace the params in the state
                trainer.train_state = full_state.replace(params=sliced_params)
                
                # Note: If your trainer stores 'n_ens' as a property, update it here:
                # trainer.model.n_ens = target_n_ens 
            else:
                trainer.train_state = full_state

            return trainer
        
        trainer_lo = restore_trainer("lo", tau_lo, 16, n_ens)
        trainer_hi = restore_trainer("hi", tau_hi, 16, n_ens)
        
        # --- Following Algorithm Steps 4-6 (Per-Channel) ---

        # 1. Get raw ensemble predictions
        y_reps_lo = get_predictions(trainer_lo, obs[cal_idx], times)
        y_reps_hi = get_predictions(trainer_hi, obs[cal_idx], times)
        y_true = gt[cal_idx] # [N_cal, res, res, res, 4]

        # Step 4: Per-function, per-replicate residuals
        # r = max(q_lo - u, u - q_hi)
        resids = jnp.maximum(y_reps_lo - y_true, y_true - y_reps_hi) 

        # Flatten only the spatial dimensions (res*res*res), keeping channels
        # New shape: [n_ens, N_cal, m_points, 4]
        n_ens, n_cal, r1, r2, r3, n_channels = resids.shape
        m_points = r1 * r2 * r3
        flat_resids = resids.reshape(n_ens, n_cal, m_points, n_channels)

        # Compute k-th order statistic (s_p,i,j) for each channel
        k = int(np.ceil((1 - alpha) * m_points))
        # Sort along the spatial dimension (axis 2) and pick k-th largest
        # Shape: [n_ens, N_cal, 4]
        s_p_i_j = jnp.sort(flat_resids, axis=2)[:, :, k - 1, :]

        # Step 5: Symmetric aggregation across replicates
        # Mean across ensemble dimension (axis 0)
        # Shape: [N_cal, 4]
        S_p_i = jnp.mean(s_p_i_j, axis=0)

        # Step 6: Calibrate the thresholds Q_p per channel
        level = (1 - alpha) * (1 + 1 / n_cal)
        # Quantile across the calibration samples (axis 0)
        Q_p = jnp.quantile(S_p_i, q=level, axis=0, method="higher")

        print(f"Calibrated thresholds per channel: {Q_p}")
        
        # 3. Data Generation & Unrolling
        gt, obs, coords, times = generate_da_data(
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

    channel_names = ['u', 'v', 'w', 'p']
    lr_suffix = f"lr{cfg.learning_rate}"
    folder_name = os.path.join(cfg.savedir, "analysis_plots_fixed_ens")
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