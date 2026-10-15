#!/bin/bash
# Docker runner for analysis_beltrami_don_pool.py. Run after
# pool_beltrami_don_calibrate_sfcqr_docker.sh has finished.
#
# Usage (from anywhere):
#   bash sbatch_scripts/beltrami/pool_beltrami_don_analysis_sfcqr_docker.sh
#   EXP_NAME=beltrami_don_sfcqr_smoke bash sbatch_scripts/beltrami/pool_beltrami_don_analysis_sfcqr_docker.sh
# Plots go to <workdir>/analysis_pool_plots/.
#
# Run inside tmux/screen (or with nohup) so it survives SSH disconnects.

set -euo pipefail

# Run from the repo root so relative paths in the config resolve
cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-beltrami_don_sfcqr}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/pool-analysis-beltrami-don-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

docker run --gpus all --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-e MPLCONFIGDIR=/tmp \
	-e JAX_LOG_LEVEL=error \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e EXP_NAME="$EXP_NAME" \
	-v "$PWD":/app -w /app \
	"$IMAGE" \
	python analysis_beltrami_don_pool.py \
	--config=configs/beltrami_don_config.py:${EXP_NAME} \
	--predict_chunk_size=8 \
	2>&1 | tee -a "$LOG"
