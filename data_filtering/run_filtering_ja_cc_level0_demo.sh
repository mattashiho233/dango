#!/bin/bash

set -e

# PLEASE UPDATE THESE PATHS TO MATCH YOUR SYSTEM
DATA_DIR="/REPLACE_HERE/ja/ja_cc/level0"
OUTPUT_DIR="/REPLACE_HERE/ja_filtered_20251118/ja_cc/level0"
UNIHAN_PATH="./unihan_bmp_no_japanese.txt"
PYTHON_SCRIPT_NAME="./filtering.py"

RUN_NAME="run_filtering_ja_cc_level0_demo"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "./logs"

python ${PYTHON_SCRIPT_NAME} \
    --data-path "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --unihan-chars-path "${UNIHAN_PATH}" \
    --num-workers 16 \
    --memory-efficient True \
    2>&1 | tee "./logs/log_${RUN_NAME}.txt"
    # --process-num-per-file 1000 \

# use --memory-efficient True when processing large datasets. 