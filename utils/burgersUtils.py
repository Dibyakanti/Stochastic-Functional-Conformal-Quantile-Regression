import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import random, vmap

import matplotlib.pyplot as plt
import pickle
import jax
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
import yaml
import os
import seaborn as sns
import pandas as pd


# def load_yaml_config(file_path):
#     with open(file_path, 'r') as file:
#         config = yaml.safe_load(file)  # Use safe_load for security
#     return config


# def save_ensemble(model, filename="ensemble.pkl"):
#     """
#     Save the entire DeepONetEnsemble including:
#       - parameters
#       - optimizer state
#       - training/test logs
#       - metadata
#     """

#     data = {
#         # Core state
#         "params": jax.tree_map(jax.device_get, model.params),
#         "opt_state": jax.tree_map(jax.device_get, model.opt_state),

#         # Logs
#         "loss_don_log": list(model.loss_don_log),
#         "loss_test_log": list(model.loss_test_log),

#         # Metadata (very useful later)
#         "n_ens": model.n_ens,
#         "loss_type": model.loss_type,
#         "alpha": model.alpha,
#         "huber_delta": model.huber_delta,
#     }

#     with open(filename, "wb") as f:
#         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# def load_ensemble(model, filename="ensemble.pkl"):
#     """
#     Load parameters and optimizer state from a single file into an existing DeepONetEnsemble.
#     """
#     with open(filename, "rb") as f:
#         data = pickle.load(f)

#     model.params = jax.tree_map(jax.device_put, data["params"])
#     model.opt_state = jax.tree_map(jax.device_put, data["opt_state"])

#     print(f"Ensemble loaded from {filename}")

def get_state(self):
    return {
        'params': self.params,
        'opt_state': self.opt_state,
        'loss_log': jnp.array(self.loss_don_log),
        'alpha': self.alpha,
        'n_ens': self.n_ens
    }

def find_files_with_string(dir_path, search_string):
    return [
        name for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
        and search_string in name
    ]


def save_checkpoint(params, filename):
    # Flatten parameters
    flat_params, unravel_fn = ravel_pytree(params)
    np.savez(filename, params=flat_params, unravel_fn=unravel_fn)


def load_checkpoint(filepath):
    data = np.load(filepath, allow_pickle=True)
    unravel_fn = data['unravel_fn'].item()
    flat_params = data['params']
    params = unravel_fn(flat_params)
    return params


def compute_single_scqr(u, y, s, ensemble_lo, ensemble_hi, q):
    # ensemble_lo_pred shape: (n_ens, num_points, 1)
    ensemble_lo_pred = vmap(lambda p: ensemble_lo.predict_u(p, u, y))(ensemble_lo.params)
    ensemble_hi_pred = vmap(lambda p: ensemble_hi.predict_u(p, u, y))(ensemble_hi.params)

    # We use a helper function to avoid passing the string directly into the vmap logic
    def get_quantile(lo, hi):
        # Calculate the conformal score for one ensemble member
        # lo/hi shape: (num_points, 1), s shape: (num_points, 1)
        diffs = jnp.array([lo - s, s - hi])
        max_val = jnp.max(diffs, axis=0)
        # We call jnp.quantile here; since this is inside the vmap, 
        # the 'method' string must be a literal or a static variable.
        return jnp.quantile(max_val, q, axis=0, method='nearest')

    # vmap over the ensemble dimension (axis 0 of the predictions)
    s_i = vmap(get_quantile)(ensemble_lo_pred, ensemble_hi_pred)

    return s_i


def compute_single(u, y, s, G ,E, q):
    # Predict for this calibration point
    E_pred = vmap(lambda p: E.predict_u(p, u, y))(E.params)  # shape (n_E,)
    G_pred = vmap(lambda p: G.predict_u(p, u, y))(G.params)  # shape (n_G,)
    
    # For each e in E_pred, compute quantile across G_pred
    h_i = vmap(
        lambda e: jnp.quantile(
            jnp.abs(s.reshape(-1, 1) - G_pred) / e,
            q,
            axis=1,
            interpolation="nearest"
        )[0, 0]
    )(E_pred)  # shape (n_E,)

    return h_i

# 3D plot function
def plot_3d(ax, X, T, f):
    surf = ax.plot_surface(X, T, f, cmap='viridis')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$u(x,t)$')


# Color plot function
def plot(ax, X, T, f):
    pcm = ax.pcolor(X, T, f, cmap='viridis')
    plt.colorbar(pcm, ax=ax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')


# Compare u(x) and s(x) if needed
def plot_us(x, u, y, s):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = '18'
    wdt = 1.5
    ax1.plot(x, u, 'k--', label='$u(x)$', linewidth=wdt)
    ax1.plot(y, s, '-', label='$s(x,t)$', linewidth=wdt)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.legend(loc='lower right', ncol=1)


# Training/test loss plots
def plot_train_test_error(model, folder_name, filename):
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    medium_purple = '#5B278F'

    # Training loss
    total_loss_eval_numbers = range(1, len(model.loss_don_log) + 1)
    axs[0].plot(total_loss_eval_numbers, model.loss_don_log, '--', color=medium_purple, label='Training loss')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'Iterations (Scaled by $10^2$)', fontsize='large')
    axs[0].set_ylabel('Training Loss', fontsize='large')
    axs[0].set_title('Evolution of Training Loss', fontsize='large')

    # Test loss
    test_loss_eval_numbers = range(1, len(model.loss_test_log) + 1)
    axs[1].plot(test_loss_eval_numbers, model.loss_test_log, '--', color=medium_purple, label='Test loss')
    axs[1].set_yscale('log')
    axs[1].set_xlabel(r'Iterations (Scaled by $10^2$)', fontsize='large')
    axs[1].set_ylabel('Test Loss', fontsize='large')
    axs[1].set_title('Evolution of Test Loss', fontsize='large')

    # Average fractional test loss
    AFTL_eval_numbers = range(1, len(model.loss_AF_test_log) + 1)
    axs[2].plot(AFTL_eval_numbers, model.loss_AF_test_log, '--', color=medium_purple, label='Avg. Fractional Test loss')
    axs[2].set_yscale('log')
    axs[2].set_xlabel(r'Iterations (Scaled by $10^2$)', fontsize='large')
    axs[2].set_ylabel('Average Fractional Test Loss', fontsize='large')
    axs[2].set_title('Evolution of Avg. Fractional Test Loss', fontsize='large')

    os.makedirs(f"./outputs/burgers/train_test/{folder_name}", exist_ok=True)
    plt.savefig(f"./outputs/burgers/train_test/{folder_name}/{filename}.png", bbox_inches="tight")


def plot_train_test_error(ensemble_lo,
                          ensemble_hi,
                          folder_name,
                          filename):
    """
    Plot training/test losses for lower and upper quantile ensembles.
    """

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    def safe_plot(ax, data, label):
        if data is not None and len(data) > 0:
            ax.plot(range(1, len(data) + 1),
                    np.array(data),
                    '--',
                    label=label)

    # =========================
    # 1. Training loss
    # =========================
    safe_plot(axs[0], getattr(ensemble_lo, "loss_don_log", None),
              "Train Loss (lo)")
    safe_plot(axs[0], getattr(ensemble_hi, "loss_don_log", None),
              "Train Loss (hi)")

    axs[0].set_yscale('log')
    axs[0].set_xlabel('Iterations (Scaled by 20)')
    axs[0].set_ylabel('Training Loss')
    axs[0].set_title('Training Loss Evolution')
    axs[0].legend()


    # =========================
    # 2. Test loss (if exists)
    # =========================
    safe_plot(axs[1], getattr(ensemble_lo, "loss_test_log", None),
              "Test Loss (lo)")
    safe_plot(axs[1], getattr(ensemble_hi, "loss_test_log", None),
              "Test Loss (hi)")

    axs[1].set_yscale('log')
    axs[1].set_xlabel('Iterations (Scaled by 20)')
    axs[1].set_ylabel('Test Loss')
    axs[1].set_title('Test Loss Evolution')
    axs[1].legend()


    # # =========================
    # # 3. Avg fractional test loss
    # # =========================
    # safe_plot(axs[2], getattr(ensemble_lo, "loss_AF_test_log", None),
    #           "AF Test Loss (lo)")
    # safe_plot(axs[2], getattr(ensemble_hi, "loss_AF_test_log", None),
    #           "AF Test Loss (hi)")

    # axs[2].set_yscale('log')
    # axs[2].set_xlabel('Iterations')
    # axs[2].set_ylabel('Average Fractional Test Loss')
    # axs[2].set_title('Avg Fractional Test Loss Evolution')
    # axs[2].legend()

    # =========================
    # Save
    # =========================
    save_dir = f"./outputs/burgers/train_test/SCQR/{folder_name}"
    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.png", bbox_inches="tight")
    plt.close(fig)

def plot_actual_pred(XX, TT, S_test, S_pred, filename):
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    # Analytical (Bateman) Solution
    axs[0, 0].tick_params(labelbottom=False, labelleft=False)
    axs[0, 0] = fig.add_subplot(221, projection='3d')
    plot_3d(axs[0, 0], XX, TT, S_test)
    axs[0, 0].set_title("Analytical (Bateman) Solution of Burgers' Equation", fontsize=16)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

    # Predicted Solution using DeepONet
    axs[0, 1].tick_params(labelbottom=False, labelleft=False)
    axs[0, 1] = fig.add_subplot(222, projection='3d')
    plot_3d(axs[0, 1], XX, TT, S_pred)
    axs[0, 1].set_title("Predicted Solution using DeepONet", fontsize=16)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

    # Bottom row: 2D color plots
    plot(axs[1, 0], XX, TT, S_test)
    axs[1, 0].set_title("Analytical (Bateman) Solution", fontsize=16)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=12)

    plot(axs[1, 1], XX, TT, S_pred)
    axs[1, 1].set_title("Predicted Solution using DeepONet", fontsize=16)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

    plt.savefig(f"./outputs/burgers/actual_predict/{filename}.png", bbox_inches="tight")


def plot_heatmap(x_list, x_label, y_list, y_label, heat_list, heat_label, folder_name, filename):
    # Create a DataFrame from your lists
    df = pd.DataFrame({
        'x': x_list,
        'y': y_list,
        'heat': heat_list
    })

    df['y_str'] = df['y'].apply(lambda x: f"{x:.3f}")
    heatmap_data = df.pivot(index='y_str', columns='x', values='heat')
    
    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': heat_label}, annot=True, fmt=".4f")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{heat_label} Heatmap')
    plt.gca().invert_yaxis()
    os.makedirs(f"./outputs/burgers/heatmap/{folder_name}/",exist_ok=True)
    plt.savefig(f"./outputs/burgers/heatmap/{folder_name}/{filename}.png", bbox_inches="tight")


def plot_line(
    x_list,
    x_label,
    y_list,
    y_label,
    line_list,
    line_label,
    folder_name,
    filename,
    draw_line=False,
    offset=0,
    shaded=False,
    draw_c_true=False,
    noise_scale=0.5
):
    # Create a DataFrame from your lists
    df = pd.DataFrame({"x": x_list, "y": y_list, "line": line_list})

    plt.figure(figsize=(8, 6))

    # Determine colormap
    if shaded:
        # Sort line values to map lighter to smaller, darker to larger
        df["line_str"] = df["line"]
        sorted_lines = sorted(df['line_str'].unique(), key=lambda s: float(s), reverse=True)
        palette = sns.color_palette("Greens_r", n_colors=len(sorted_lines))
        color_map = dict(zip(sorted_lines, palette))
    else:
        df["line_str"] = df["line"].apply(lambda x: f"{x:.3f}")
        palette = sns.color_palette(n_colors=df['line_str'].nunique())
        color_map = dict(zip(df['line_str'].unique(), palette))

    # Plot the main lines
    ax = sns.lineplot(
        data=df, x="x", y="y", hue="line_str", marker="o", palette=color_map
    )

    if draw_line:
        for line_str, color in color_map.items():
            y_val = df[df['line_str'] == line_str]['line_str'].iloc[-1]
            ax.axhline(y=float(y_val)+offset, color=color, linestyle=':', linewidth=1)
            
    if draw_c_true:
        plt.yscale("log")
        thetas = jnp.linspace(0.9, float(max(df["y"])/2), 1000000)
        # sample_num = 1000000
        # samples = noise_scale * random.laplace(random.PRNGKey(123), shape=(sample_num,))
        # y = vmap(lambda theta: jnp.average(jnp.abs(samples) <= theta))(thetas)
        b = noise_scale
        y = 1.0 - jnp.exp(-thetas / b)
        plt.plot(y, 2*thetas, color="black", linewidth=1, label=r'$C_{true}$')
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label} for each {line_label}')
    plt.legend(title=line_label)
    os.makedirs(f"./outputs/burgers/lineplot/{folder_name}/",exist_ok=True)
    plt.savefig(f"./outputs/burgers/lineplot/{folder_name}/{filename}.png", bbox_inches="tight")