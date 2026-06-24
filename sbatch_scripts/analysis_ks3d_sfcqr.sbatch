#!/bin/bash --login

#SBATCH -p gpuA
#SBATCH --gpus=1
#SBATCH -t 0-2
#SBATCH --output=./outputs/slurm/analysis-ks3d-%j.out

module purge
module load libs/cuda

module avail 2>&1 | grep -i xvfb
module avail 2>&1 | grep -i x11
module avail 2>&1 | grep -i xorg



# export SINGULARITYENV_JAX_LOG_LEVEL=error
# export SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=2

export APPTAINERENV_JAX_LOG_LEVEL=error
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2


export EXP_NAME=${EXP_NAME:-baseline}
# export SINGULARITYENV_EXP_NAME="$EXP_NAME"
export APPTAINERENV_EXP_NAME="$EXP_NAME"

export SINGULARITY_HOME="$HOME"
export SINGULARITY_LANG="$LANG"

export SINGULARITY_BINDPATH="/scratch,/mnt"

# PyVista/VTK off-screen rendering
export APPTAINERENV_PYVISTA_OFF_SCREEN=true
export APPTAINERENV_VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1
export APPTAINERENV_MPLBACKEND=Agg


# if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
# 	export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
# fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi


singularity exec --nv ./singularity/jax-csf3_latest.sif python ./analysis_sfcqr.py --config="configs/config.py:${EXP_NAME}"

# singularity exec --nv ./singularity/jax-csf3_latest.sif \
#     xvfb-run -a -s "-screen 0 1920x1080x24" \
#     python ./analysis_sfcqr.py --config="configs/config.py:${EXP_NAME}"