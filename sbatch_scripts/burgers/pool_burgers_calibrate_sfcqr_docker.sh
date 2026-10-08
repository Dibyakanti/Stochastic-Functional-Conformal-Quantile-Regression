#!/bin/bash
# Docker equivalent of pool_burgers_calibrate_sfcqr.sbatch. Run after
# pool_burgers_sfcqr_docker.sh has finished training the pool.
#
# Usage (from anywhere):
#   bash sbatch_scripts/burgers/pool_burgers_calibrate_sfcqr_docker.sh
#   EXP_NAME=burgers_sfcqr_d5 bash sbatch_scripts/burgers/pool_burgers_calibrate_sfcqr_docker.sh
# Sizes larger than the pool are skipped (5 batches x 16 = 80 members).
#
# Run inside tmux/screen (or with nohup) so it survives SSH disconnects.

set -euo pipefail

# Run from the repo root so relative paths in the config resolve
cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-burgers_sfcqr}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/pool-calibrate-burgers-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

docker run --gpus all --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-e JAX_LOG_LEVEL=error \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e EXP_NAME="$EXP_NAME" \
	-v "$PWD":/app -w /app \
	"$IMAGE" \
	python calibrate_burgers_pool.py \
	--config=configs/burgers_config.py:${EXP_NAME} \
	--n_ens_sub_list=1,2,4,6,8,10,16,32 \
	--n_repeats=5 \
	2>&1 | tee -a "$LOG"
