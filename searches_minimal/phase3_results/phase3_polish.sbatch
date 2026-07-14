#!/bin/bash
#SBATCH --job-name=p3_polish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/ral/jnightin/phase3_logs/%x-%j.out
#SBATCH --error=/mnt/ral/jnightin/phase3_logs/%x-%j.err

set -u
export PYTHONUNBUFFERED=1
source /mnt/ral/jnightin/autolens_profiling/activate.sh
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR=/mnt/ral/jnightin/scratch/jax_cache
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd /mnt/ral/jnightin/autolens_workspace_developer
echo "===== NODE / GPU ====="; hostname; nvidia-smi -L
python -c "import jax; print(chr(39)+chr(39), jax.default_backend(), jax.devices())"
export MULTISTART_N_STARTS=${N:-128}
echo "########## RUN AdamLBFGS $(date +%H:%M:%S) ##########"
python -m searches_minimal.adam_then_lbfgs
echo "########## DONE AdamLBFGS rc=$? $(date +%H:%M:%S) ##########"
