#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -t 0-2
#SBATCH --output=./outputs/slurm/animations-%j.out

# GIF of the SFCQR band for any preset (see animations.py).
#   EXP_NAME=sfcqr_beltrami_adamw sbatch sbatch_scripts/animations.sbatch
#   EXP_NAME=burgers_sfcqr_d3 CONFIG_FILE=configs/burgers_config.py sbatch sbatch_scripts/animations.sbatch
#   EXP_NAME=sfcqr_beltrami_adamw ANIM_ARGS="--channel=u --name=beltrami_uvit" sbatch sbatch_scripts/animations.sbatch

module purge

export SINGULARITYENV_JAX_LOG_LEVEL=error
export SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=2

export EXP_NAME=${EXP_NAME:-baseline}
export CONFIG_FILE=${CONFIG_FILE:-configs/config.py}

export SINGULARITY_HOME="$HOME"
export SINGULARITY_LANG="$LANG"
export SINGULARITY_BINDPATH="/scratch,/mnt"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

# pyvista renders off screen through Xvfb (installed in the image).
unset DISPLAY

singularity exec --nv ./singularity/jax-csf3_latest.sif python animations.py \
  --config=${CONFIG_FILE}:${EXP_NAME} \
  --batch_n_ens=${BATCH_N_ENS:-16} \
  --predict_chunk_size=${PREDICT_CHUNK_SIZE:-8} \
  ${ANIM_ARGS}
