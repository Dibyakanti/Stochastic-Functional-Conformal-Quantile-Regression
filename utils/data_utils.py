import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Enable 64-bit precision for accurate physics simulation
jax.config.update("jax_enable_x64", True)

def dataloader(
    ground_truth: jax.Array, 
    observations: jax.Array, 
    times: jax.Array, 
    batch_size: int, 
    shuffle: bool = True
):
    """
    Slices a (time, X, Y, Z, C) trajectory into (x, y) pairs for DA.
    x: state at t
    y: state at t+dt
    """
    num_time_steps = observations.shape[0]
    # We have (T-1) possible pairs
    indices = np.arange(num_time_steps - 1)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < batch_size:
                continue
                
            # 'x' is the noisy observation at time t
            # 'y' is the ground truth (or noisy observation) at time t+dt
            yield {
                "x": observations[batch_idx],
                "y": observations[batch_idx + 1],
                "time": times[batch_idx]
            }

def da_dataloader(
    indices: jax.Array, 
    ground_truth: jax.Array, 
    observations: jax.Array, 
    times: jax.Array, 
    batch_size: int, 
    mode: str = "train"  # "train" for odd, "eval" for even
):
    # num_time_steps = observations.shape[0]
    # # Indices 0 to T-2 (to allow for i+1 target)
    # all_indices = np.arange(num_time_steps - 3)
    
    # if mode == "train":
    #     # Keep only odd indices: t=1, 3, 5...
    #     indices = all_indices[all_indices % 2 != 0]
    # else:
    #     # Keep only even indices starting from t=2: t=2, 4, 6...
    #     indices = all_indices[(all_indices % 2 == 0) & (all_indices >= 2)]
    
    indices_np = np.array(indices)
    
    while True:
        np.random.shuffle(indices_np)
        for i in range(0, len(indices_np), batch_size):
            batch_idx = indices_np[i:i + batch_size]
            if len(batch_idx) < batch_size: continue
            yield {
                "x": observations[batch_idx],
                "y": observations[batch_idx + 2],
                "time": times[batch_idx + 2] - times[batch_idx]
            }