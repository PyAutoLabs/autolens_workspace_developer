#!/bin/bash
#SBATCH --job-name=p3_secondorder
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
echo "########## RUN LM $(date +%H:%M:%S) ##########"
python -m searches_minimal.jaxopt_lm_multistart
echo "########## DONE LM rc=$? $(date +%H:%M:%S) ##########"
echo "########## RUN GN $(date +%H:%M:%S) ##########"
python -m searches_minimal.jaxopt_gaussnewton_multistart
echo "########## DONE GN rc=$? $(date +%H:%M:%S) ##########"
echo "########## RUN LBFGS $(date +%H:%M:%S) ##########"
python -m searches_minimal.jaxopt_lbfgs_multistart
echo "########## DONE LBFGS rc=$? $(date +%H:%M:%S) ##########"
echo "########## RUN BFGS $(date +%H:%M:%S) ##########"
python -m searches_minimal.jaxopt_bfgs_multistart
echo "########## DONE BFGS rc=$? $(date +%H:%M:%S) ##########"
echo "########## RUN NCG $(date +%H:%M:%S) ##########"
python -m searches_minimal.jaxopt_nonlinear_cg
echo "########## DONE NCG rc=$? $(date +%H:%M:%S) ##########"
