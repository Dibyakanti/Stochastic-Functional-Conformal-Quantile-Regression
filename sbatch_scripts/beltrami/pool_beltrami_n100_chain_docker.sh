#!/bin/bash
# Train pool -> calibrate -> analysis for sfcqrclaude_beltrami_adamw with
# ~100 train / 100 cal / 100 test samples (time_steps=410, dt=0.02 keeps the
# same t in [0, 8] horizon as the base preset), on a single-GPU machine.
#
# Usage (re-running resumes: batches with a final checkpoint are skipped):
#   nohup setsid bash sbatch_scripts/beltrami/pool_beltrami_n100_chain_docker.sh \
#       > outputs/docker/chain-beltrami-n100.out 2>&1 &

set -uo pipefail

# Run from the repo root so relative paths in the config resolve
cd "$(dirname "$0")/../.."

export EXP_NAME=${EXP_NAME:-sfcqrclaude_beltrami_adamw}
IMAGE=${IMAGE:-dibyakanti/jax-csf3:latest}
WORKDIR=./results/sfcqrclaude_beltrami_adamw_n100
STEPS=4000
OVR="--config=configs/config.py:${EXP_NAME} --config.time_steps=410 --config.dt=0.02 --config.num_unroll_steps=100 --config.total_train_steps=${STEPS} --config.workdir=${WORKDIR}"

mkdir -p ./outputs/docker
TS=$(date +%Y%m%d-%H%M%S)
DR="docker run --gpus all --rm --user $(id -u):$(id -g) -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e JAX_LOG_LEVEL=error -e TF_CPP_MIN_LOG_LEVEL=2 -v $PWD:/app -w /app $IMAGE"

for i in $(seq 0 9); do
	# Last (alpha, quantile) trained in a batch is alpha_0.3/hi
	if [ -d "${WORKDIR}/alpha_0.3/hi/batch_${i}/checkpoints/${STEPS}" ]; then
		echo "== batch_idx=$i already done, skipping"
		continue
	fi
	$DR python train_ensemble_pool.py $OVR --batch_idx=$i --batch_n_ens=16 \
		|| echo "!! batch_idx=$i failed (exit $?), continuing"
done 2>&1 | tee -a ./outputs/docker/pool-beltrami-${EXP_NAME}_n100-${TS}.out

$DR python calibrate_sfcqr.py $OVR --batch_n_ens=16 \
	--n_ens_sub_list=1,2,4,6,8,10,12,14,16 --n_repeats=5 \
	2>&1 | tee -a ./outputs/docker/calibrate-beltrami-${EXP_NAME}_n100-${TS}.out

$DR python analysis_pool.py $OVR --batch_n_ens=16 --predict_chunk_size=8 \
	2>&1 | tee -a ./outputs/docker/analysis-beltrami-${EXP_NAME}_n100-${TS}.out
echo CHAIN_DONE
