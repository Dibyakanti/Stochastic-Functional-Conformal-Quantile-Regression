"""Configs for the pooled DeepONet/Beltrami SFCQR pipeline.

Select a preset with --config=configs/beltrami_don_config.py:<name>, e.g.
  --config=configs/beltrami_don_config.py:beltrami_don_sfcqr

The data is the same as the UNet Beltrami runs (`generate_beltrami_da_data`):
one function per time index i, with input the noisy field obs[i] and target the
noisy field obs[i + 2]. The branch net is the UViT encoder of `UNet3d` (time
embedding + DStack, then a linear head) applied to the whole field obs[i]; the
trunk is an MLP on (x, y, z). A single DeepONet outputs all four channels
(u, v, w, p).

Data values (res, time_steps, dt, noise_std, seed) match
`presets.sfcqrclaude_smallnn_beltrami_adamw`, so the train/cal split is the
same set of time indices as the UNet run.
"""

from ml_collections import config_dict


def beltrami_don_sfcqr():
    config = config_dict.ConfigDict()
    config.type = "beltrami"
    config.seed = 1234
    config.workdir = "./results/beltrami_uvit_don_sfcqr"

    # Data (as in the UNet preset)
    config.res = 16
    config.time_steps = 80
    config.dt = 0.1
    config.noise_std = 2
    # Test rollout, as in the UNet analysis: inputs obs[2 + 2i], truth
    # gt[4 + 2i], for i < num_unroll_steps.
    config.num_unroll_steps = 20

    # Train: P_train fixed random grid points per training function. Each
    # step a member sees fn_batch_size functions x pts_per_fn of their points.
    config.P_train = 2000
    config.nIter = 3000
    config.fn_batch_size = 4
    config.pts_per_fn = 256

    # SFCQR
    config.alpha_list = [0.20, 0.25, 0.30]
    config.lr_list = [1e-3]
    config.beta = 0.20
    config.variant = "A"

    config.channels = ["u", "v", "w", "p"]
    config.model = config_dict.ConfigDict({
        # UNet3d encoder settings (as in presets.sfcqrclaude_beltrami_adamw;
        # the rest are UNet3d defaults). res must be divisible by the product
        # of downsample_ratio.
        "branch": {
            "num_channels": (16, 32, 64),
            "downsample_ratio": (2, 2, 2),
            "use_spatial_attention": (False, False, True),
            "num_blocks": 4,
            "noise_embed_dim": 128,
            "input_proj_channels": 128,
            "num_heads": 8,
        },
        # The first entry is replaced by 3 (x, y, z); the last is the latent
        # size p per channel, so both nets output 4 * p.
        "trunk_layers": [3, 512, 256, 128, 64],
        "loss_type": "quantile",
    })
    return config


def beltrami_don_sfcqr_smoke():
    """Tiny run for checking the pipeline end to end."""
    config = beltrami_don_sfcqr()
    config.P_train = 100
    config.nIter = 50
    config.alpha_list = [0.20]
    config.workdir = "./results/beltrami_uvit_don_sfcqr_smoke"
    return config


def get_config(variant="beltrami_don_sfcqr"):
    options = {
        "beltrami_don_sfcqr": beltrami_don_sfcqr,
        "beltrami_don_sfcqr_smoke": beltrami_don_sfcqr_smoke,
    }
    return options[variant]()
