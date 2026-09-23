#!/bin/bash
#SBATCH --job-name=weaver-train
#SBATCH --partition=NV_H100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --time=168:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:?submit with --export=ALL,DATA_ROOT=/path/to/h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/weaver}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/weaver.yaml}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"
RESUME="${RESUME:-none}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export WEAVER_RESUME="${RESUME}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi
srun "${PYTHON_CMD[@]}" train.py --config "${CONFIG}" \
  --data.root_dir "${DATA_ROOT}" \
  --trainer.default_root_dir "${OUTPUT_ROOT}"
