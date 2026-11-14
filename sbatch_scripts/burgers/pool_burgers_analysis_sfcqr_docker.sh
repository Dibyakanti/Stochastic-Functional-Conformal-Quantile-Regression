#!/bin/bash
# Docker equivalent of pool_analysis_burgers_calibrate_sfcqr.sbatch. Run after
# pool_burgers_calibrate_sfcqr_docker.sh has finished.
#
# Usage (from anywhere):
#   bash sbatch_scripts/burgers/pool_burgers_analysis_sfcqr_docker.sh
#   EXP_NAME=burgers_sfcqr_d5 bash sbatch_scripts/burgers/pool_burgers_analysis_sfcqr_docker.sh
# Plots go to <workdir>/analysis_pool_plots/.
#
# Run inside tmux/screen (or with nohup) so it survives SSH disconnects.

set -euo pipefail

# Run from the repo root so relative paths in the config resolve
cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-burgers_sfcqr}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/pool-analysis-burgers-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

docker run --gpus all --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-e MPLCONFIGDIR=/tmp \
	-e JAX_LOG_LEVEL=error \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e EXP_NAME="$EXP_NAME" \
	-v "$PWD":/app -w /app \
	"$IMAGE" \
	python analysis_burgers_pool.py \
	--config=configs/burgers_config.py:${EXP_NAME} \
	--predict_chunk_size=8 \
	2>&1 | tee -a "$LOG"
