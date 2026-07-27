#!/bin/bash
#SBATCH --job-name=pix_prodigy_cpu
#SBATCH --partition=ral
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/ral/jnightin/pixgrad_logs/%x-%j.out
#SBATCH --error=/mnt/ral/jnightin/pixgrad_logs/%x-%j.err

# Pixelized MultiStartProdigy campaign (autolens_workspace_developer#117):
# one (rule, mesh) chain link per submission. The search persists its
# search_internal every 50 steps under a fixed name/path_prefix, so
# re-submitting the SAME (PIX_MODE, PIX_MESH, PIX_N_STEPS) resumes from the
# last persisted step — chain 24 h jobs until the fixed step budget completes
# or the fom_history plateaus (the #101 wall signature).
#
#   sbatch --export=ALL,PIX_MESH=rectangular searches_minimal/pix_prodigy_cpu.sbatch
#   sbatch --export=ALL,PIX_MESH=knn        searches_minimal/pix_prodigy_cpu.sbatch
#   sbatch --export=ALL,PIX_MESH=delaunay   searches_minimal/pix_prodigy_cpu.sbatch
#   PIX_MODE=nautilus ... -> the per-mesh converged-sampler baseline instead.
#
# CPU is the point here (A100s busy; #117 wants "does it reach the right
# solution and in how many steps", not wall-time). ~5-6 min/step at 16 starts
# (job 330953); memory bounded by PIX_BATCH tiling (Fit#1374).

set -u
export PYTHONUNBUFFERED=1
source /mnt/ral/jnightin/autolens_profiling/activate.sh
# No GPU on the `ral` partition: pin JAX to CPU so it does not error probing CUDA.
export JAX_PLATFORMS=cpu
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
# Warm compile cache: the first pix compile is ~35 min cold, seconds warm.
export JAX_COMPILATION_CACHE_DIR=/mnt/ral/jnightin/scratch/jax_cache
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd /mnt/ral/jnightin/autolens_workspace_developer

# JAX_ENABLE_X64 is NOT inherited by sbatch environments — x64 is set inside
# the script (pix_prodigy.py), never rely on the submit shell for it.
export PIX_MESH="${PIX_MESH:-rectangular}"
export PIX_N_STEPS="${PIX_N_STEPS:-3000}"
export PIX_BATCH="${PIX_BATCH:-4}"
MODE="${PIX_MODE:-prodigy}"

echo "===== NODE ====="; hostname
python -c "import jax; print('backend', jax.default_backend(), jax.devices())"

echo ""
echo "########## ${MODE} mesh=${PIX_MESH}  $(date +%H:%M:%S) ##########"
python -m searches_minimal.pix_prodigy "$MODE"
rc=$?
echo "########## DONE rc=${rc}  $(date +%H:%M:%S) ##########"
exit $rc
