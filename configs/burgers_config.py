"""Configs for the pooled DeepONet/Burgers SFCQR pipeline.

Select a preset with --config=configs/burgers_config.py:<name>, e.g.
  --config=configs/burgers_config.py:burgers_sfcqr

Values come from UncertaintyQuantification_OperatorNets/configs/presets_sfcqr.py;
beta/variant match the UNet SFCQR presets.
"""

from ml_collections import config_dict


def burgers_sfcqr():
    config = config_dict.ConfigDict()
    config.type = "burgers"
    config.seed = 1234
    config.workdir = "./results/burgers_sfcqr"

    # Train: N_train functions (half held out for calibration), P_train random
    # (x, t) points per function.
    config.N_train = 400
    config.P_train = 200
    config.nIter = 2000
    config.noise_scale = 1.0
    config.batch_size = 50

    # Test: N_test functions on a P_test x P_test grid.
    config.N_test = 100
    config.P_test = 20

    # SFCQR
    config.alpha_list = [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**-0.5]
    # One lr for every quantile. In a validation sweep on held-out training
    # functions, lr >= 1e-2 had members diverge at the extreme quantiles
    # (tau = 1/64, 63/64); 5e-3 at 2000 iterations was within 7% of the best
    # lr at all 12 quantiles used here.
    config.lr_list = [5e-3]
    config.beta = 0.20
    config.variant = "A"

    config.model = config_dict.ConfigDict({
        "branch_layers": [100, 64],
        "trunk_layers": [2, 64],
        "loss_type": "quantile",
    })
    config.grid_data = config_dict.ConfigDict({
        "Nx": 100,
        "Nt": 100,
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
    return options[variant]()
