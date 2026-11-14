#!/bin/bash --login
# Train the ensemble pool with one Slurm array task per batch (batch_idx =
# SLURM_ARRAY_TASK_ID), so batches run in parallel and a failed/timed-out
# batch can be resubmitted on its own (e.g. --array=3,7).
#
# Usage (from the repo root):
#   EXP_NAME=sfcqr_beltrami_adamw_n100 sbatch sbatch_scripts/beltrami/pool_beltrami_array_sfcqr.sbatch
#
# The split (split.npz in the workdir) must already exist, or be created by a
# single batch first, to avoid tasks racing to create it.

#SBATCH --account gpu-af-insilico
#SBATCH -p gpuH
#SBATCH --gpus=1
#SBATCH -t 0-12
#SBATCH --array=0-9
#SBATCH --output=./outputs/slurm/pool-beltrami-%A_%a.out

module purge

export SINGULARITYENV_JAX_LOG_LEVEL=error
export SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=2

export EXP_NAME=${EXP_NAME:-baseline}
export SINGULARITYENV_EXP_NAME="$EXP_NAME"

export SINGULARITY_HOME="$HOME"
export SINGULARITY_LANG="$LANG"

export SINGULARITY_BINDPATH="/scratch,/mnt"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

singularity exec --nv ./singularity/jax-csf3_latest.sif python train_ensemble_pool.py \
  --config=configs/config.py:${EXP_NAME} \
  --batch_idx=${SLURM_ARRAY_TASK_ID} \
  --batch_n_ens=16
