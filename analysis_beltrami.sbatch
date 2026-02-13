#!/bin/bash --login

#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -t 1-0
#SBATCH --output=./outputs/slurm/analysis-beltrami-%j.out


export EXP_NAME=${EXP_NAME:-baseline}
export SINGULARITYENV_EXP_NAME="$EXP_NAME"

module purge
module load libs/cuda

export SINGULARITY_HOME="$HOME"
export SINGULARITY_LANG="$LANG"

export SINGULARITY_BINDPATH="/scratch,/mnt"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
	export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

singularity exec --nv ./singularity/jax-csf3_latest.sif python ./analysis.py --config="configs/config.py:${EXP_NAME}"