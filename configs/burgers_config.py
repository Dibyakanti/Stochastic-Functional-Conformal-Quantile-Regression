"""Configs for the pooled DeepONet/Burgers SFCQR pipeline.

Select a preset with --config=configs/burgers_config.py:<name>, e.g.
  --config=configs/burgers_config.py:burgers_sfcqr

The data is the d-dimensional viscous Burgers solution
  u(x, t) = 1 / (1 + exp((sum_i x_i - d t / 2) / (2 nu))),  x in R^d,
with one function per viscosity nu. Append `_d<N>` to any preset name to set
the spatial dimension (default 1), e.g. `burgers_sfcqr_d5`. The workdir always
ends in `_d<dim>`, so runs at different dimensions never share checkpoints.

Values come from UncertaintyQuantification_OperatorNets/configs/presets_sfcqr.py;
beta/variant match the UNet SFCQR presets.
"""

import re

from ml_collections import config_dict


def burgers_sfcqr():
    config = config_dict.ConfigDict()
    config.type = "burgers"
    config.seed = 1234
    config.workdir = "./results/burgers_sfcqr"
    # Spatial dimension d. Use the `_d<N>` preset suffix rather than
    # --config.dim, so the workdir follows it.
    config.dim = 1

    # Train: N_train functions (half held out for calibration), P_train random
    # (x, t) points per function.
    config.N_train = 400
    config.P_train = 200
    config.nIter = 1500
    config.noise_scale = 1.0
    config.batch_size = 50

    # Test: N_test functions, each at the same P_test**2 random (x, t) points.
    config.N_test = 100
    config.P_test = 20

    # SFCQR
    # config.alpha_list = [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**-0.5]
    config.alpha_list = [0.20, 0.25, 0.30]
    # One lr for every quantile. In a validation sweep on held-out training
    # functions, lr >= 1e-2 had members diverge at the extreme quantiles
    # (tau = 1/64, 63/64); 5e-3 at 2000 iterations was within 7% of the best
    # lr at all 12 quantiles used here.
    config.lr_list = [5e-3]
    config.beta = 0.20
    config.variant = "A"

    # branch_layers[0] is the number of initial-condition sensors. The first
    # entry of trunk_layers is replaced by dim + 1 (x_1..x_d, t) at build time.
    config.model = config_dict.ConfigDict({
        "branch_layers": [100, 128, 64],
        "trunk_layers": [2, 128, 64],
        "loss_type": "quantile",
    })
    # Points are drawn from [xmin, xmax]^dim x [tmin, tmax]; nu is drawn from
    # [vmin, vmax).
    config.grid_data = config_dict.ConfigDict({
        "xmin": -5,
        "xmax": 5,
        "tmin": 0,
        "tmax": 10,
        "vmin": 2**7,
        "vmax": 2**9,
    })
    return config


def burgers_sfcqr_variantB():
    config = burgers_sfcqr()
    config.variant = "B"
    config.workdir = "./results/burgers_sfcqr_variantB"
    return config


def burgers_sfcqr_smoke():
    """Tiny run (a few minutes on CPU) for checking the pipeline end to end."""
    config = burgers_sfcqr()
    config.N_train = 40
    config.P_train = 50
    config.N_test = 10
    config.nIter = 50
    config.alpha_list = [2**-3]
    config.workdir = "./results/burgers_sfcqr_smoke"
    return config


def get_config(variant="burgers_sfcqr"):
    options = {
        "burgers_sfcqr": burgers_sfcqr,
        "burgers_sfcqr_variantB": burgers_sfcqr_variantB,
        "burgers_sfcqr_smoke": burgers_sfcqr_smoke,
    }
    m = re.fullmatch(r"(.+)_d(\d+)", variant)
    name, dim = (m.group(1), int(m.group(2))) if m else (variant, 1)
    config = options[name]()
    config.dim = dim
    config.workdir = f"{config.workdir}_d{dim}"
    return config
