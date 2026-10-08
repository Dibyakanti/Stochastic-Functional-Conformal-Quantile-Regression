# UViT
This repository contains the code for performing data assimilation using UViT.

Commands for running in CSF3:
1. `EXP_NAME=sfcqr sbatch beltrami_sfcqr.sbatch`

Comands for running in RONIN:
1. `docker pull dibyakanti/jax-csf3:latest`
2. `tmux new -s <some_name>`
2. `bash sbatch_scripts/burgers/pool_burgers_sfcqr_docker.sh`