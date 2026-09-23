#!/bin/bash
#SBATCH --job-name=weaver-validate
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:?submit with --export=ALL,DATA_ROOT=/path/to/h5}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_ROOT}/outputs/release_validation}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/weaver_weights.ckpt}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "${ARTIFACT_DIR}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=("${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output python)
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi

srun "${PYTHON_CMD[@]}" -m compileall -q \
  train.py infer.py predict.py evaluate.py weaver scripts tests
srun "${PYTHON_CMD[@]}" -m pytest -q tests
shopt -s nullglob
H5_FILES=("${DATA_ROOT}"/test/*.h5)
H5_INPUT="${H5_FILES[0]:-}"
if [[ -z "${H5_INPUT}" ]]; then
  echo "no test HDF5 file found below ${DATA_ROOT}/test" >&2
  exit 2
fi
srun "${PYTHON_CMD[@]}" scripts/export_h5_input.py --h5 "${H5_INPUT}" \
  --output "${ARTIFACT_DIR}/physical_input.npy"
srun "${PYTHON_CMD[@]}" predict.py --input "${ARTIFACT_DIR}/physical_input.npy" \
  --output "${ARTIFACT_DIR}/forecast_6h.npy" --lead-time 6 \
  --checkpoint "${CHECKPOINT}"
srun "${PYTHON_CMD[@]}" evaluate.py --data-root "${DATA_ROOT}" \
  --ckpt "${CHECKPOINT}" --lead-times 6 --batch-size 1 --max-batches 1 \
  --output "${ARTIFACT_DIR}/metrics_smoke.json"
