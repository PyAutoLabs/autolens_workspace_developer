#!/bin/bash
#SBATCH --job-name=smc_warm_cpu
#SBATCH --partition=ral
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/ral/jnightin/smc_grad_logs/%x-%j.out
#SBATCH --error=/mnt/ral/jnightin/smc_grad_logs/%x-%j.err

set -u
export PYTHONUNBUFFERED=1

source /mnt/ral/jnightin/autolens_profiling/activate.sh

# Modern blackjax (1.5) shadowing the venv's ancient 0.1.0b1.dev.
export PYTHONPATH="/mnt/ral/jnightin/scratch/smc_grad_pylibs:${PYTHONPATH}"

export JAX_PLATFORMS=cpu
# CRITICAL: float64 (jax defaults to float32; not inherited by sbatch).
export JAX_ENABLE_X64=True
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR=/mnt/ral/jnightin/scratch/jax_cache
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"

cd /mnt/ral/jnightin/autolens_workspace_developer

echo "===== NODE ====="; hostname; nproc; free -g | head -2
python -c "import jax; print('backend', jax.default_backend(), 'x64', jax.config.jax_enable_x64)"
python -c "import blackjax; print('blackjax', blackjax.__version__)"

# ---------------------------------------------------------------------------
# STEP 1: warm start. Multi-start Prodigy -> cached MLE + reference scale.
# SANITY CHECK the printed mle: einstein_radius (index 8) should land near the
# truth 1.6 (see _grad_setup einstein_truth); if it is far off, Prodigy has not
# converged and every downstream warm arm is meaningless.
# ---------------------------------------------------------------------------
echo ""
echo "########## STEP 1: multi-start Prodigy warm start  $(date +%H:%M:%S) ##########"
python -m searches_minimal._warm_start --n-starts 12 --n-steps 300 --refresh \
  || echo "WARMSTART rc=$? (downstream arms will fail)"
echo "########## STEP 1 DONE  $(date +%H:%M:%S) ##########"

# ---------------------------------------------------------------------------
# STEP 2/3: the representative regime — temper from the normalised Gaussian
# reference centred on that MLE (log Z stays valid), fixed auto-step then tuned.
# ---------------------------------------------------------------------------
echo ""
echo "########## STEP 2: SMC warm-start (auto step from reference width)  $(date +%H:%M:%S) ##########"
python -m searches_minimal.blackjax_smc_grad \
  --warm-start --kernel mala --n-particles 64 --num-mcmc-steps 3 --target-ess 0.6 \
  || echo "ARM-WARM rc=$? (continuing)"
echo "########## STEP 2 DONE  $(date +%H:%M:%S) ##########"

echo ""
echo "########## STEP 3: SMC warm-start + tuning  $(date +%H:%M:%S) ##########"
python -m searches_minimal.blackjax_smc_grad \
  --warm-start --tune --kernel mala --n-particles 64 --num-mcmc-steps 3 --target-ess 0.6 \
  || echo "ARM-WARM-TUNE rc=$? (continuing)"
echo "########## STEP 3 DONE  $(date +%H:%M:%S) ##########"

echo ""
echo "===== SMC WARM-START RUN COMPLETE $(date +%H:%M:%S) ====="
