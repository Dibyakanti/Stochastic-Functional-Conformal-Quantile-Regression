from ml_collections import config_dict

def baseline():
    config = config_dict.ConfigDict()
    config.res = 64
    config.batch_size = 8
    config.learning_rate = 1e-4
    config.total_train_steps = 4000
    config.noise_std = 0
    config.seed = 0
    config.workdir = "./results/baseline"
    config.model = config_dict.ConfigDict({
        'num_channels': (32, 64, 128, 256),
        'use_spatial_attention': (False, False, True, True)
    })
    return config

def high_res():
    config = baseline()
    config.res = 128
    config.batch_size = 4
    config.workdir = "./results/high_res"
    return config