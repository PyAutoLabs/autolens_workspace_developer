#!/bin/bash
#SBATCH --job-name=pix_slogdet_ab_cpu
#SBATCH --partition=ral
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/ral/jnightin/pixgrad_logs/%x-%j.out
#SBATCH --error=/mnt/ral/jnightin/pixgrad_logs/%x-%j.err

# CPU sibling of pix_slogdet_ab.sbatch (autolens_workspace_developer#112).
# The A100 arm exists only for throughput; this A/B is a *correctness* question
# — does the shipped gradient-safe slogdet log-det (PyAutoArray#392) reopen the
# pixelized multi-start gradient landscape that cholesky NaN-walled in #100/#101?
# The verdict signals (step-0 finite-gradient acceptance + per-start "died after
# step N") do not need GPU speed, and the `ral` CPU partition has idle ~900 GB /
# 252-CPU nodes with a 27-day walltime. Memory (10.9 GiB/point, batched to stay
# bounded) is the only hard requirement and 128 GB covers it comfortably.
# Both arms run the SAME script, SAME seed(0), one env var (PIX_LOGDET) apart:
#   - baseline: PIX_LOGDET unset -> "cholesky", the NaN-walling arm
#   - fix:      PIX_LOGDET=slogdet, finite + differentiable where cholesky NaNs

set -u
export PYTHONUNBUFFERED=1
source /mnt/ral/jnightin/autolens_profiling/activate.sh
# No GPU on the `ral` partition: pin JAX to CPU so it does not error probing CUDA.
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR=/mnt/ral/jnightin/scratch/jax_cache
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd /mnt/ral/jnightin/autolens_workspace_developer

# CPU is materially slower than the A100, but the verdict lives in the early
# steps (cholesky NaN-walls by ~step 25-50; slogdet must survive past that), so
# the default step count is trimmed from 300. Override PIX_N_STEPS for a full
# convergence comparison. PIX_N_STARTS/PIX_BATCH keep their script defaults.
export PIX_N_STEPS="${PIX_N_STEPS:-150}"

echo "===== NODE ====="; hostname
python -c "import jax; print('backend', jax.default_backend(), jax.devices())"
python -c "import autoarray as aa; print('log_det_method supported:', 'log_det_method' in aa.Settings.__init__.__code__.co_varnames)"

RULE="${PIX_RULE:-adam}"

echo ""
echo "########## ARM 1: BASELINE cholesky  $(date +%H:%M:%S) ##########"
env -u PIX_LOGDET python -m searches_minimal.pix_lr_free "$RULE"
echo "########## DONE cholesky rc=$?  $(date +%H:%M:%S) ##########"
cp searches_minimal/output/pix_lr_free_comparison.txt \
   searches_minimal/output/pix_lr_free_cholesky_cpu_${SLURM_JOB_ID}.txt 2>/dev/null || true

echo ""
echo "########## ARM 2: slogdet  $(date +%H:%M:%S) ##########"
PIX_LOGDET=slogdet python -m searches_minimal.pix_lr_free "$RULE"
echo "########## DONE slogdet rc=$?  $(date +%H:%M:%S) ##########"
cp searches_minimal/output/pix_lr_free_comparison.txt \
   searches_minimal/output/pix_lr_free_slogdet_cpu_${SLURM_JOB_ID}.txt 2>/dev/null || true

echo ""
echo "===== A/B COMPLETE $(date +%H:%M:%S) ====="
