#!/bin/bash
# Docker equivalent of sbatch_scripts/ks3d/pool_analysis_ks3d_calibrate_sfcqr.sbatch
# for the Challenge 2.1 data.
#
# Usage (from anywhere):
#   bash sbatch_scripts/challenge/pool_challenge_analysis_sfcqr_docker.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-sfcqr_smallnn_challenge_adamw}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/analysis-challenge-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

docker run --gpus all --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-e MPLCONFIGDIR=/tmp \
	-e JAX_LOG_LEVEL=error \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e EXP_NAME="$EXP_NAME" \
	-v "$PWD":/app -w /app \
	"$IMAGE" \
	python analysis_pool.py \
	--config=configs/config.py:${EXP_NAME} \
	--batch_n_ens=16 \
	--predict_chunk_size=8 \
	2>&1 | tee -a "$LOG"
