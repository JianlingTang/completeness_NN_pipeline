#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -N comp_pipeline_photo
#PBS -j oe
#PBS -m bea
#PBS -l storage=scratch/jh2+gdata/jh2+scratch/mk27
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/comp_pipeline.o
#PBS -e /scratch/jh2/jt4478/output/comp_pipeline.e

set -euo pipefail

# ============ 只填路径 / FILL PATHS BELOW ============
# 项目根（解压 zip 的目录，所有生成结果写在这里）
PROJECT_ROOT="/g/data/jh2/jt4478/make_LC_copy"

# 输入数据路径（可不在项目根下）
COMP_SLUG_LIB_DIR="/g/data/jh2/jt4478/SLUG_library"
COMP_PSF_PATH="/g/data/jh2/jt4478/PSF_files"
COMP_FITS_PATH="/g/data/jh2/jt4478/ngc628_data"
COMP_SCIFRAME=""   # 留空则用 COMP_FITS_PATH/ngc628-c/ngc628-c_white.fits；否则填白光 FITS 完整路径
COMP_BAO_PATH="/g/data/jh2/jt4478/.deps/local/bin"
IRAF="/path/to/iraf"   # IRAF 安装目录，做 photometry 必填

# 可选：Python 虚拟环境（若用 venv 且已装在项目根下）
# VENV="${PROJECT_ROOT}/.venv/bin/activate"
VENV=""
# ============ 路径结束 / END PATHS ============

module load python3/3.11.0
[[ -n "${VENV}" && -f "${VENV}" ]] && source "${VENV}"

export COMP_SLUG_LIB_DIR
export COMP_PSF_PATH
export COMP_FITS_PATH
export COMP_BAO_PATH
export IRAF
[[ -n "${COMP_SCIFRAME}" ]] && export COMP_SCIFRAME

cd "${PROJECT_ROOT}"

python scripts/run_pipeline.py \
  --cleanup \
  --nframe 3 \
  --reff_list "1,3,5,7,9" \
  --ncl 500 \
  --run_photometry \
  --parallel \
  --n_workers 8
