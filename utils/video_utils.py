import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vape4d import render
from vape4d.utils import diverging_alpha

def stack_time_series(vol_list):
    """
    Convert list of (res,res,res) → (T,res,res,res)
    """
    return np.stack(vol_list, axis=0)


def save_video(video, path, fps=10):
    """
    Save vape4d output frames as MP4
    """
    fig = plt.figure()
    plt.axis("off")

    frames = []
    for i in range(len(video)):
        frames.append([plt.imshow(video[i], animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=1000/fps, blit=True)

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(path, writer=writer)

    plt.close(fig)
    print(f"saved → {path}")