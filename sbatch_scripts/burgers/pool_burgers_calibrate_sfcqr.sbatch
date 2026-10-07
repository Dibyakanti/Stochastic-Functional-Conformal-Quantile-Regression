#!/bin/bash --login

# SFCQR calibration over the whole pool. Run after all training tasks finish:
#   EXP_NAME=burgers_sfcqr sbatch --dependency=afterok:<train_job_id> \
#       sbatch_scripts/burgers/pool_burgers_calibrate_sfcqr.sbatch
# Sizes larger than the pool are skipped (10 batches x 8 = 80 members).

#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -t 0-2
#SBATCH --output=./outputs/slurm/pool-calibrate-burgers-%j.out

module purge
# module load libs/cuda

export SINGULARITYENV_JAX_LOG_LEVEL=error
export SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=2

export EXP_NAME=${EXP_NAME:-burgers_sfcqr}
export SINGULARITYENV_EXP_NAME="$EXP_NAME"

export SINGULARITY_HOME="$HOME"
export SINGULARITY_LANG="$LANG"

export SINGULARITY_BINDPATH="/scratch,/mnt"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi


singularity exec --nv ./singularity/jax-csf3_latest.sif python calibrate_burgers_pool.py \
  --config=configs/burgers_config.py:${EXP_NAME} \
  --n_ens_sub_list=1,2,4,6,8,10,16,32 \
  --n_repeats=5