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
    config.workdir = "./results/sfcqr_smallnn_beltrami_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : (2, 2)
    })
    return config

def sfcqr_smallnn_beltrami_adamw_trainlonger():
    config = baseline()
    config.type="beltrami"
    config.optimizer="adamw"
    config.res = 32
    config.batch_size = 4
    config.learning_rate = 1e-3
    config.total_train_steps = 2000
    config.time_steps = 80
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_beltrami_adamw_trainlonger"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
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
    config.total_train_steps = 4000
    config.time_steps = 80
    config.noise_std = 2
    config.n_ens = 2
    config.workdir = "./results/sfcqr_beltrami_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, False, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_beltrami_adamw_domain5():
    """sfcqr_beltrami_adamw on [-5, 5]^3 instead of [-1, 1]^3; nothing else
    changes (raw, unnormalised fields; noise_std = 2 as in the base run)."""
    config = sfcqr_beltrami_adamw()
    config.domain = 5.0
    config.workdir = "./results/sfcqr_beltrami_adamw_domain5"
    return config

def sfcqr_beltrami_adamw_n100():
    # ~100 train / 100 cal / 100 test samples. dt=0.02 with 410 steps keeps
    # the same t in [0, 8] horizon as sfcqr_beltrami_adamw.
    # Matches sbatch_scripts/beltrami/pool_beltrami_n100_chain_docker.sh.
    config = sfcqr_beltrami_adamw()
    config.time_steps = 410
    config.dt = 0.02
    config.num_unroll_steps = 100
    config.total_train_steps = 4000
    config.workdir = "./results/sfcqr_beltrami_adamw_n100"
    config.model = config_dict.ConfigDict({
        'num_channels': (32, 64, 128),
        'use_spatial_attention': (False, True, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_smallnn_ks_adamw():
    config = baseline()
    config.type="ks3d"
    config.optimizer="adamw"
    config.res = 16
    config.batch_size = 8
    config.learning_rate = 1e-3
    config.total_train_steps = 2000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_ks_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
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
    config.batch_size = 32
    config.learning_rate = 1e-3
    config.total_train_steps = 2000
    config.time_steps = 102
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_ks_adamw"
    config.num_unroll_steps = 20
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    # config.alpha_list = [0.20, 0.25, 0.30]
    # config.alpha_list = [2**-5, 2**-4, 2**-3]
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, False, True),
        'downsample_ratio' : (2, 2, 2)
    })
    return config

def sfcqr_smallnn_challenge_adamw():
    config = baseline()
    config.type="challenge2_1"
    config.data_dir = "./dataset"
    config.optimizer="adamw"
    config.res = None  # grid is (152, 48, 1), set by the loader
    config.batch_size = 8
    config.learning_rate = 1e-3
    config.total_train_steps = 2000
    config.time_steps = 400
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_smallnn_challenge_adamw"
    config.num_unroll_steps = 199
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32),
        'use_spatial_attention': (False, False),
        'downsample_ratio' : ((2, 2, 1),) * 2,
        # z is a singleton axis: (3, 3, 1) makes every conv an exact 2D conv
        'kernel_size': (3, 3, 1),
    })
    return config

def sfcqr_challenge_adamw():
    config = baseline()
    config.type="challenge2_1"
    config.data_dir = "./dataset"
    config.optimizer="adamw"
    config.res = None  # grid is (152, 48, 1), set by the loader
    config.batch_size = 8
    config.learning_rate = 1e-3
    config.total_train_steps = 2000
    config.time_steps = 400
    config.noise_std = 0.0
    config.n_ens = 2
    config.workdir = "./results/sfcqr_challenge_adamw"
    config.num_unroll_steps = 199
    config.alpha_list = [0.20, 0.25, 0.30]
    config.beta = 0.20
    config.n_ens_list = [1, 2, 4, 8, 16]
    config.variant = "A"
    config.model = config_dict.ConfigDict({
        'num_channels': (16, 32, 64),
        'use_spatial_attention': (False, False, True),
        'downsample_ratio' : ((2, 2, 1),) * 3,
        # z is a singleton axis: (3, 3, 1) makes every conv an exact 2D conv
        'kernel_size': (3, 3, 1),
    })
    return config

def sfcqr_smallnn_challenge2_2_adamw():
    config = sfcqr_smallnn_challenge_adamw()
    config.type = "challenge2_2"
    config.case = "a25/f0p2"  # one trajectory: a<mean angle>/f<freq>
    config.workdir = "./results/sfcqr_smallnn_challenge2_2_adamw"
    return config

def sfcqr_challenge2_2_adamw():
    config = sfcqr_challenge_adamw()
    config.type = "challenge2_2"
    config.case = "a25/f0p2"  # one trajectory: a<mean angle>/f<freq>
    config.workdir = "./results/sfcqr_challenge2_2_adamw"
    return config
