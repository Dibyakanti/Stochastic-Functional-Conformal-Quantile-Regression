from ml_collections import config_dict

def baseline():
    config = config_dict.ConfigDict()
    config.res = 64
    config.batch_size = 8
    config.learning_rate = 1e-4
    config.total_train_steps = 4000
    config.time_steps = 50
    config.dt = 0.1
    config.num_unroll_steps = 5
    config.noise_std = 0
    config.seed = 0
    config.workdir = "./results/baseline"
    config.model = config_dict.ConfigDict({
        'num_channels': (32, 64, 128, 256),
        'use_spatial_attention': (False, False, True, True),
        'downsample_ratio' : (2, 2, 2, 2)
    })
    return config

def sfcqr():
    config = baseline()
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-4
    config.total_train_steps = 1000
    config.time_steps = 50
    config.n_ens = 2
    config.workdir = "./results/sfcqr"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 6]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_lower_lr():
    config = baseline()
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-2
    config.total_train_steps = 1000
    config.time_steps = 50
    config.n_ens = 2
    config.workdir = "./results/sfcqr_lower_lr"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    config.n_ens_list = [1, 2, 4, 6]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def high_res():
    config = baseline()
    config.res = 128
    config.batch_size = 4
    config.workdir = "./results/high_res"
    return config

def noisy_in_out():
    config = baseline()
    config.batch_size = 16
    config.noise_std = 1.5
    config.workdir = "./results/noisy_in_out"
    return config