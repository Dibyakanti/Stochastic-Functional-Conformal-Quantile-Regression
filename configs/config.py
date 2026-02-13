from configs import presets

def get_config(variant="baseline"):
    
    options = {
        "baseline": presets.baseline(),
        "high_res": presets.high_res(),
    }
    return options[variant]