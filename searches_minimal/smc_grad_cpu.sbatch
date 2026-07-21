#!/bin/bash
#SBATCH --job-name=smc_grad_cpu
#SBATCH --partition=ral
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/ral/jnightin/smc_grad_logs/%x-%j.out
#SBATCH --error=/mnt/ral/jnightin/smc_grad_logs/%x-%j.err

set -u
export PYTHONUNBUFFERED=1

# PyAuto library PYTHONPATH + base venv (jax 0.10.2 CPU).
source /mnt/ral/jnightin/autolens_profiling/activate.sh

# The base venv ships an ancient blackjax (0.1.0b1.dev); prepend the isolated
# modern install (1.5, matching local) so it shadows the old one on sys.path.
export PYTHONPATH="/mnt/ral/jnightin/scratch/smc_grad_pylibs:${PYTHONPATH}"

# No GPU on the `ral` partition: pin JAX to CPU so it does not error probing CUDA.
export JAX_PLATFORMS=cpu
# CRITICAL: float64. The MGE inversion gradients (cholesky/log-det) are
# precision-sensitive and the gradient certification (probe_grad.py) is float64;
# jax defaults to float32 unless this is set, so it MUST be exported explicitly
# here (it is ambient in the local shell but not inherited by the batch job).
export JAX_ENABLE_X64=True
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR=/mnt/ral/jnightin/scratch/jax_cache
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"

cd /mnt/ral/jnightin/autolens_workspace_developer

echo "===== NODE ====="; hostname; nproc; free -g | head -2
echo "===== JAX / blackjax ====="
python -c "import jax; print('backend', jax.default_backend(), jax.devices())"
python -c "import blackjax; print('blackjax', blackjax.__version__)"

# Convergence check (not representative timing — CPU). 128 particles is ample to
# judge whether the gradient SMC anneals into the high-likelihood region and
# produces a sensible log Z vs the Nautilus baseline (~ -169k smoke / nss_grad
# ~ -31). Arms are independent: a failure in one must not abort the other.

# Graph kept small (64 particles, 3 MCMC steps) so the vmapped+scanned inversion
# gradient does not blow XLA/LLVM compile memory (the 128p/8-step graph did).

echo ""
echo "########## ARM 1: MALA + inner_kernel_tuning (spread-adaptive step)  $(date +%H:%M:%S) ##########"
python -m searches_minimal.blackjax_smc_grad \
  --kernel mala --tune --n-particles 64 --num-mcmc-steps 3 --step-size 0.1 --target-ess 0.6 \
  || echo "ARM1 rc=$? (continuing)"
echo "########## ARM 1 DONE  $(date +%H:%M:%S) ##########"

echo ""
echo "########## ARM 2: MALA fixed tiny-step control (0.001)  $(date +%H:%M:%S) ##########"
python -m searches_minimal.blackjax_smc_grad \
  --kernel mala --n-particles 64 --num-mcmc-steps 3 --step-size 0.001 --target-ess 0.6 \
  || echo "ARM2 rc=$? (continuing)"
echo "########## ARM 2 DONE  $(date +%H:%M:%S) ##########"

echo ""
echo "===== SMC-GRAD CPU CONVERGENCE RUN COMPLETE $(date +%H:%M:%S) ====="
