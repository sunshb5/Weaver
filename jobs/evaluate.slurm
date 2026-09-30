#!/bin/bash
#SBATCH --job-name=weaver-evaluate
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:?submit with --export=ALL,DATA_ROOT=/path/to/h5,OUTPUT=/path/to/metrics.json}"
OUTPUT="${OUTPUT:?submit with --export=ALL,DATA_ROOT=/path/to/h5,OUTPUT=/path/to/metrics.json}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/weaver_weights.ckpt}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"
LEAD_TIMES="${LEAD_TIMES:-24 72 120 240}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
read -r -a LEADS <<< "${LEAD_TIMES}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi
srun "${PYTHON_CMD[@]}" evaluate.py --data-root "${DATA_ROOT}" --output "${OUTPUT}" \
  --ckpt "${CHECKPOINT}" --lead-times "${LEADS[@]}"
