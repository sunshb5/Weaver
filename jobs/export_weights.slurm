#!/bin/bash
#SBATCH --job-name=weaver-export
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
INPUT="${INPUT:?full training checkpoint path is required}"
OUTPUT="${OUTPUT:?new inference checkpoint path is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_BIN="${CONDA_BIN:-conda}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi
srun "${PYTHON_CMD[@]}" scripts/export_weights.py \
  --input "${INPUT}" --output "${OUTPUT}"

