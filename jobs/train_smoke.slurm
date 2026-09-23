#!/bin/bash
#SBATCH --job-name=weaver-train-smoke
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:?submit with --export=ALL,DATA_ROOT=/path/to/h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/train_smoke}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/weaver.yaml}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi

# fast_dev_run exercises model construction, one optimizer step, and the
# validation loop without creating a release checkpoint.
srun "${PYTHON_CMD[@]}" train.py --config "${CONFIG}" \
  --data.root_dir "${DATA_ROOT}" \
  --data.batch_size 1 --data.val_batch_size 1 --data.num_workers 0 \
  --trainer.strategy auto --trainer.devices 1 --trainer.num_nodes 1 \
  --trainer.fast_dev_run true --trainer.default_root_dir "${OUTPUT_ROOT}"
