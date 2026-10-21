#!/bin/bash
# Docker equivalent of sbatch_scripts/ks3d/pool_ks3d_calibrate_sfcqr.sbatch
# for the Challenge 2.1 data.
#
# Usage (from anywhere):
#   bash sbatch_scripts/challenge/pool_challenge_calibrate_sfcqr_docker.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-sfcqrclaude_smallnn_challenge_adamw}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}

mkdir -p ./outputs/docker
LOG=./outputs/docker/calibrate-challenge-${EXP_NAME}-$(date +%Y%m%d-%H%M%S).out

docker run --gpus all --rm \
	--user "$(id -u):$(id -g)" \
	-e HOME=/tmp \
	-e JAX_LOG_LEVEL=error \
	-e TF_CPP_MIN_LOG_LEVEL=2 \
	-e EXP_NAME="$EXP_NAME" \
	-v "$PWD":/app -w /app \
	"$IMAGE" \
	python calibrate_sfcqr.py \
	--config=configs/config.py:${EXP_NAME} \
	--batch_n_ens=16 \
	--n_ens_sub_list=1,2,4,6,8,10,12,14,16 \
	--n_repeats=5 \
	2>&1 | tee -a "$LOG"
