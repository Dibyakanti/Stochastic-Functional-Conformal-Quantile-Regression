from configs import presets

def get_config(variant="baseline"):
    
    options = {
        "baseline": presets.baseline(),
        "sfcqr_smallnn_ks_adamw": presets.sfcqr_smallnn_ks_adamw(),
        "sfcqr_ks_adamw": presets.sfcqr_ks_adamw(),
        "sfcqr_smallnn_beltrami_adamw_trainlonger": presets.sfcqr_smallnn_beltrami_adamw_trainlonger(),
        "sfcqr_smallnn_beltrami_adamw": presets.sfcqr_smallnn_beltrami_adamw(),
        "sfcqr_beltrami_adamw": presets.sfcqr_beltrami_adamw(),
        "sfcqr_beltrami_adamw_n100": presets.sfcqr_beltrami_adamw_n100(),
        "sfcqr_smallnn_challenge_adamw": presets.sfcqr_smallnn_challenge_adamw(),
        "sfcqr_challenge_adamw": presets.sfcqr_challenge_adamw(),
        "sfcqr_smallnn_challenge2_2_adamw": presets.sfcqr_smallnn_challenge2_2_adamw(),
        "sfcqr_challenge2_2_adamw": presets.sfcqr_challenge2_2_adamw()
    }
    return options[variant]