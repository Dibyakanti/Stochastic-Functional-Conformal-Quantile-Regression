from configs import presets

def get_config(variant="baseline"):
    
    options = {
        "baseline": presets.baseline(),
        "sfcqr": presets.sfcqr(),
        "sfcqr_lower_lr": presets.sfcqr_lower_lr(),
        "high_res": presets.high_res()
    }
    return options[variant]