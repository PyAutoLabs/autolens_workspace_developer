#!/bin/bash
#SBATCH --job-name=pix_slogdet_ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ral/jnightin/pixgrad_logs/%x-%j.out
#SBATCH --error=/mnt/ral/jnightin/pixgrad_logs/%x-%j.err

# A/B validation of the shipped gradient-safe slogdet log-det (PyAutoArray#392)
# on the pixelized multi-start gradient landscape (autolens_workspace_developer#112).
# Both arms run the SAME script, SAME seed(0), one env var (PIX_LOGDET) apart:
#   - baseline: PIX_LOGDET unset -> "cholesky", the arm that NaN-walled in #100/#101
#   - fix:      PIX_LOGDET=slogdet, finite + differentiable where the Cholesky NaNs
# The verdict signals are all in stdout: the "Collected X/N finite-gradient starts
# (from T draws)" line (step-0 NaN-grad acceptance) and the per-start "died after
# step N" death report (mid-trajectory walls).

set -u
export PYTHONUNBUFFERED=1
source /mnt/ral/jnightin/autolens_profiling/activate.sh
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR=/mnt/ral/jnightin/scratch/jax_cache
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd /mnt/ral/jnightin/autolens_workspace_developer

echo "===== NODE / GPU ====="; hostname; nvidia-smi -L
python -c "import jax; print('backend', jax.default_backend(), jax.devices())"
python -c "import autoarray as aa; print('log_det_method supported:', 'log_det_method' in aa.Settings.__init__.__code__.co_varnames)"

RULE="${PIX_RULE:-adam}"

echo ""
echo "########## ARM 1: BASELINE cholesky  $(date +%H:%M:%S) ##########"
env -u PIX_LOGDET python -m searches_minimal.pix_lr_free "$RULE"
echo "########## DONE cholesky rc=$?  $(date +%H:%M:%S) ##########"
cp searches_minimal/output/pix_lr_free_comparison.txt \
   searches_minimal/output/pix_lr_free_cholesky_${SLURM_JOB_ID}.txt 2>/dev/null || true

echo ""
echo "########## ARM 2: slogdet  $(date +%H:%M:%S) ##########"
PIX_LOGDET=slogdet python -m searches_minimal.pix_lr_free "$RULE"
echo "########## DONE slogdet rc=$?  $(date +%H:%M:%S) ##########"
cp searches_minimal/output/pix_lr_free_comparison.txt \
   searches_minimal/output/pix_lr_free_slogdet_${SLURM_JOB_ID}.txt 2>/dev/null || true

echo ""
echo "===== A/B COMPLETE $(date +%H:%M:%S) ====="
