#!/bin/bash
#SBATCH --job-name=weaver-predict
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
INPUT="${INPUT:?submit with --export=ALL,INPUT=/path/to/physical_input.npy,OUTPUT=/path/to/forecast.npy}"
OUTPUT="${OUTPUT:?submit with --export=ALL,INPUT=/path/to/physical_input.npy,OUTPUT=/path/to/forecast.npy}"
LEAD_TIME="${LEAD_TIME:-24}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/weaver_weights.ckpt}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi
srun "${PYTHON_CMD[@]}" predict.py --input "${INPUT}" --output "${OUTPUT}" \
  --lead-time "${LEAD_TIME}" --checkpoint "${CHECKPOINT}"
