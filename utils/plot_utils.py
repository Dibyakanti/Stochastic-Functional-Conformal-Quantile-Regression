import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

def _plot_unrolled_intervals(lo, hi, truth, plot_path):
    """Plots the time series of the unrolled interval for a central point."""
    mid = lo.shape[1] // 2
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(truth))
    
    plt.fill_between(steps, lo[:, mid, mid, mid, 0], hi[:, mid, mid, mid, 0], 
                     color='blue', alpha=0.2, label="Calibrated Interval")
    plt.plot(steps, truth[:, mid, mid, mid, 0], 'k-', label="Ground Truth")
    plt.xlabel("Unroll Steps")
    plt.ylabel("U-Velocity")
    plt.title("Autoregressive Unrolling: 90% Calibrated Prediction Interval")
    plt.legend()
    plt.savefig(os.path.join(plot_path, "unrolled_coverage.png"))
    plt.close()
    
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
    noise_scale=0.5,
    y_scale = None
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
        b = noise_scale
        y = 1.0 - jnp.exp(-thetas / b)
        plt.plot(y, 2*thetas, color="black", linewidth=1, label=r'$C_{true}$')
        
    if y_scale=="log":
        plt.yscale("log")
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label} for each {line_label}')
    plt.legend(title=line_label)
    os.makedirs(f"{folder_name}/",exist_ok=True)
    plt.savefig(f"{folder_name}/{filename}.png", bbox_inches="tight")