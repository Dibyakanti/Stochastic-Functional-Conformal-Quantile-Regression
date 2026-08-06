from ml_collections import config_dict

def baseline():
    config = config_dict.ConfigDict()
    config.optimizer = "adamw"
    config.ens_type = "mean"
    config.optim_noise = 1.0
    config.res = 64
    config.batch_size = 8
    config.learning_rate = 1e-4
    config.total_train_steps = 4000
    config.time_steps = 60
    config.dt = 0.1
    config.num_unroll_steps = 5
    config.noise_std = 0.0
    config.seed = 1234
    config.workdir = "./results/baseline"
    config.model = config_dict.ConfigDict({
        'num_channels': (32, 64, 128, 256),
        'use_spatial_attention': (False, False, True, True),
        'downsample_ratio' : (2, 2, 2, 2)
    })
    return config

def sfcqr():
    config = baseline()
    config.type="beltrami"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-4
    config.total_train_steps = 1000
    config.time_steps = 60
    config.n_ens = 2
    config.noise_std = 2
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

def sfcqr_fixedens():
    config = baseline()
    config.type="beltrami"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 60
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_smallnn"
    config.savedir = "./results/sfcqr_noisy_fixedens"
    config.num_unroll_steps = 20
    # config.alpha_list = [0.30]
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16, 32]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_highres():
    config = baseline()
    config.type="beltrami"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-4
    config.total_train_steps = 1000
    config.time_steps = 60
    config.n_ens = 2
    config.workdir = "./results/sfcqr_highres"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 6]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_random_smallnn_beltrami_adamw():
    config = baseline()
    config.type="beltrami"
    config.optimizer="adamw"
    config.ens_type = "random"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 60
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_random_noisy_smallnn_beltrami_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_smallnn_beltrami_adamw():
    config = baseline()
    config.type="beltrami"
    config.optimizer="adamw"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 60
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_smallnn_beltrami_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_beltrami_adamw():
    config = baseline()
    config.type="beltrami"
    config.optimizer="adamw"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_beltrami_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_smallnn_beltrami_asgld():
    config = baseline()
    config.type="beltrami"
    config.optimizer="asgld"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 80
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_smallnn_beltrami_asgld"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_random_smallnn_beltrami_asgld():
    config = baseline()
    config.type="beltrami"
    config.optimizer="asgld"
    config.ens_type = "random"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 80
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_random_noisy_smallnn_beltrami_asgld"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_beltrami_asgld():
    config = baseline()
    config.type="beltrami"
    config.optimizer="asgld"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 6
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_beltrami_asgld"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_smallnn_beltrami_asgld_lesstemp():
    config = baseline()
    config.type="beltrami"
    config.optimizer="asgld"
    config.optim_noise = 1e-2
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 80
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_smallnn_beltrami_asgld_lesstemp"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.25]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [1, 2, 4, 8]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_random_smallnn_beltrami_asgld_lesstemp():
    config = baseline()
    config.type="beltrami"
    config.optimizer="asgld"
    config.ens_type = "random"
    config.optim_noise = 1e-2
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 80
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_random_noisy_smallnn_beltrami_asgld_lesstemp"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.25]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [1, 2, 4, 8]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_higher_lr():
    config = baseline()
    config.type="beltrami"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-2
    config.total_train_steps = 1000
    config.time_steps = 60
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_higher_lr"
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
    config.type="beltrami"
    config.res = 128
    config.noise_std = 2
    config.batch_size = 4
    config.workdir = "./results/high_res"
    return config

def noisy_in_out():
    config = baseline()
    config.type="beltrami"
    config.batch_size = 16
    config.noise_std = 1.5
    config.workdir = "./results/noisy_in_out"
    return config

# # # # # # # # # # # # # # # # 
#   Kuramoto-Shivanisky 3D    # 
# # # # # # # # # # # # # # # # 

def sfcqr_smallnn_ks_adamw():
    config = baseline()
    config.type="ks3d"
    config.optimizer="adamw"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_ks_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_random_smallnn_ks_adamw():
    config = baseline()
    config.type="ks3d"
    config.optimizer="adamw"
    config.ens_type = "random"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_random_smallnn_ks_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_ks_adamw():
    config = baseline()
    config.type="ks3d"
    config.optimizer="adamw"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_ks_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_smallnn_ks_adamwnoise():
    config = baseline()
    config.type="ks3d"
    config.optimizer="adamwnoise"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_noisy_smallnn_ks_adamwnoise"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_smallnn_ks_asgld():
    config = baseline()
    config.type="ks3d"
    config.optimizer="asgld"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_ks_asgld"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_random_smallnn_ks_asgld():
    config = baseline()
    config.type="ks3d"
    config.optimizer="asgld"
    config.ens_type="random"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_random_smallnn_ks_asgld"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_smallnn_ks_asgld_temp2_5():
    config = baseline()
    config.type="ks3d"
    config.optimizer="asgld"
    config.optim_noise = 2**-5
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_ks_asgld_temp2_5"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_smallnn_ks_asgld_lesstemp():
    config = baseline()
    config.type="ks3d"
    config.optimizer="asgld"
    config.optim_noise = 1e-2
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_ks_asgld_lesstemp"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_random_smallnn_ks_asgld_lesstemp():
    config = baseline()
    config.type="ks3d"
    config.optimizer="asgld"
    config.ens_type="random"
    config.optim_noise = 1e-2
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_random_smallnn_ks_asgld_lesstemp"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_smallnn_ks_adamsgld():
    config = baseline()
    config.type="ks3d"
    config.optimizer="adamsgld"
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_ks_adamsgld"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_ks_asgld_temp2_5():
    config = baseline()
    config.type="ks3d"
    config.optimizer="asgld"
    config.optim_noise = 2**-5
    config.res = 16
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 1000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_ks_asgld_temp2_5"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8]
    # config.n_ens_list = [16]
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config