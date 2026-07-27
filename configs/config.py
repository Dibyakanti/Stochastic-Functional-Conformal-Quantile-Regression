from configs import presets

def get_config(variant="baseline"):
    
    options = {
        "baseline": presets.baseline(),
        "sfcqr": presets.sfcqr(),
        "sfcqr_smallnn_beltrami_adamw": presets.sfcqr_smallnn_beltrami_adamw(),
        "sfcqr_random_smallnn_beltrami_adamw": presets.sfcqr_random_smallnn_beltrami_adamw(),
        "sfcqr_beltrami_adamw": presets.sfcqr_beltrami_adamw(),
        "sfcqr_smallnn_beltrami_asgld": presets.sfcqr_smallnn_beltrami_asgld(),
        "sfcqr_random_smallnn_beltrami_asgld": presets.sfcqr_random_smallnn_beltrami_asgld(),
        "sfcqr_beltrami_asgld": presets.sfcqr_beltrami_asgld(),
        "sfcqr_random_smallnn_beltrami_asgld_lesstemp": presets.sfcqr_random_smallnn_beltrami_asgld_lesstemp(),
        "sfcqr_smallnn_beltrami_asgld_lesstemp": presets.sfcqr_smallnn_beltrami_asgld_lesstemp(),
        "sfcqr_higher_lr": presets.sfcqr_higher_lr(),
        "sfcqr_highres": presets.sfcqr_highres(),
        "sfcqr_fixedens": presets.sfcqr_fixedens(),
        "high_res": presets.high_res(),
        "sfcqr_smallnn_ks_adamw": presets.sfcqr_smallnn_ks_adamw(),
        "sfcqr_random_smallnn_ks_adamw": presets.sfcqr_random_smallnn_ks_adamw(),
        "sfcqr_ks_adamw": presets.sfcqr_ks_adamw(),
        "sfcqr_smallnn_ks_adamwnoise": presets.sfcqr_smallnn_ks_adamwnoise(),
        "sfcqr_smallnn_ks_asgld": presets.sfcqr_smallnn_ks_asgld(),
        "sfcqr_random_smallnn_ks_asgld": presets.sfcqr_random_smallnn_ks_asgld(),
        "sfcqr_smallnn_ks_asgld_lesstemp": presets.sfcqr_smallnn_ks_asgld_lesstemp(),
        "sfcqr_random_smallnn_ks_asgld_lesstemp": presets.sfcqr_random_smallnn_ks_asgld_lesstemp(),
        "sfcqr_smallnn_ks_asgld_temp2_5": presets.sfcqr_smallnn_ks_asgld_temp2_5(),
        "sfcqr_smallnn_ks_adamsgld": presets.sfcqr_smallnn_ks_adamsgld(),
        "sfcqr_ks_asgld_temp2_5": presets.sfcqr_ks_asgld_temp2_5()
    }
    return options[variant]