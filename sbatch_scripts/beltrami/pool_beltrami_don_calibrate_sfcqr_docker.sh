#!/bin/bash
# Docker runner for calibrate_beltrami_don_pool.py. Run after
# pool_beltrami_don_sfcqr_docker.sh has finished training the pool.
#
# Usage (from anywhere):
#   bash sbatch_scripts/beltrami/pool_beltrami_don_calibrate_sfcqr_docker.sh
#   EXP_NAME=beltrami_don_sfcqr_smoke bash sbatch_scripts/beltrami/pool_beltrami_don_calibrate_sfcqr_docker.sh
# Sizes larger than the pool are skipped (5 batches x 16 = 80 members).
#
# Run inside tmux/screen (or with nohup) so it survives SSH disconnects.

set -euo pipefail

# Run from the repo root so relative paths in the config resolve
cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-beltrami_don_sfcqr}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/pool-calibrate-beltrami-don-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

docker run --gpus all --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-e JAX_LOG_LEVEL=error \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e EXP_NAME="$EXP_NAME" \
	-v "$PWD":/app -w /app \
	"$IMAGE" \
	python calibrate_beltrami_don_pool.py \
	--config=configs/beltrami_don_config.py:${EXP_NAME} \
	--n_ens_sub_list=1,2,4,6,8,10,16,32 \
	--n_repeats=5 \
	2>&1 | tee -a "$LOG"
