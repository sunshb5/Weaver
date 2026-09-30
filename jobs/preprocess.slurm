#!/bin/bash
#SBATCH --job-name=weaver-data
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
STEP="${STEP:?set STEP to download, regrid, h5, stats, clim, or regions}"
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

case "${STEP}" in
  download)
    SOURCE="${SOURCE:?SOURCE must name the WeatherBench2 Zarr dataset or URI}"
    RAW_DIR="${RAW_DIR:?RAW_DIR is required}"
    srun "${PYTHON_CMD[@]}" -m \
      weaver.data_preprocessing.download_wb2 --file "${SOURCE}" --save_dir "${RAW_DIR}"
    ;;
  regrid)
    RAW_DIR="${RAW_DIR:?RAW_DIR is required}"
    REGRID_DIR="${REGRID_DIR:?REGRID_DIR is required}"
    srun "${PYTHON_CMD[@]}" -m \
      weaver.data_preprocessing.regrid_wb2 --root_dir "${RAW_DIR}" --save_dir "${REGRID_DIR}"
    ;;
  h5)
    REGRID_DIR="${REGRID_DIR:?REGRID_DIR is required}"
    DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
    SPLIT="${SPLIT:?SPLIT must be train, val, or test}"
    case "${SPLIT}" in
      train) START_YEAR=2008; END_YEAR=2016 ;;
      val) START_YEAR=2017; END_YEAR=2017 ;;
      test) START_YEAR=2018; END_YEAR=2018 ;;
      *) echo "invalid SPLIT=${SPLIT}" >&2; exit 2 ;;
    esac
    srun "${PYTHON_CMD[@]}" -m \
      weaver.data_preprocessing.process_one_step_data \
      --root_dir "${REGRID_DIR}" --save_dir "${DATA_ROOT}" --split "${SPLIT}" \
      --start_year "${START_YEAR}" --end_year "${END_YEAR}"
    ;;
  stats)
    REGRID_DIR="${REGRID_DIR:?REGRID_DIR is required}"
    STATISTICS_DIR="${STATISTICS_DIR:-${PROJECT_ROOT}/assets/statistics}"
    for LEAD in -1 6 12 24; do
      srun "${PYTHON_CMD[@]}" -m \
        weaver.data_preprocessing.compute_normalization \
        --root_dir "${REGRID_DIR}" --save_dir "${STATISTICS_DIR}" \
        --start_year 2008 --end_year 2016 --lead_time "${LEAD}"
    done
    ;;
  clim)
    DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
    STATISTICS_DIR="${STATISTICS_DIR:-${PROJECT_ROOT}/assets/statistics}"
    srun "${PYTHON_CMD[@]}" \
      scripts/compute_climatology.py --data-dir "${DATA_ROOT}/train" \
      --output "${STATISTICS_DIR}/clim.npz"
    ;;
  regions)
    STATISTICS_DIR="${STATISTICS_DIR:-${PROJECT_ROOT}/assets/statistics}"
    srun "${PYTHON_CMD[@]}" \
      scripts/build_region_map.py --clim "${STATISTICS_DIR}/clim.npz"
    ;;
  *) echo "invalid STEP=${STEP}" >&2; exit 2 ;;
esac
