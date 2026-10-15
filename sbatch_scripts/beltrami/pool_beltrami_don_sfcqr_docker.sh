#!/bin/bash
# Docker runner for train_beltrami_don_pool.py (DeepONet on Beltrami), for a
# single-GPU machine (e.g. Ronin) without Slurm/Singularity.
#
# Usage (from anywhere):
#   bash sbatch_scripts/beltrami/pool_beltrami_don_sfcqr_docker.sh
#   EXP_NAME=beltrami_don_sfcqr_smoke bash sbatch_scripts/beltrami/pool_beltrami_don_sfcqr_docker.sh
#
# Run inside tmux/screen (or with nohup) so it survives SSH disconnects.

set -euo pipefail

# Run from the repo root so relative paths in the config resolve
cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-beltrami_don_sfcqr}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/pool-beltrami-don-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

for i in $(seq 0 4); do
	docker run --gpus all --rm \
		--user "$(id -u):$(id -g)" \
		-e HOME=/tmp \
		-e JAX_LOG_LEVEL=error \
		-e TF_CPP_MIN_LOG_LEVEL=2 \
		-e EXP_NAME="$EXP_NAME" \
		-v "$PWD":/app -w /app \
		"$IMAGE" \
		python train_beltrami_don_pool.py \
		--config=configs/beltrami_don_config.py:${EXP_NAME} \
		--batch_idx=$i \
		--batch_n_ens=16 \
		|| echo "!! batch_idx=$i failed (exit $?), continuing"
done 2>&1 | tee -a "$LOG"
