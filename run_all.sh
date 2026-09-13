#!/usr/bin/env bash
# One-click HeuFouFT pipeline: data preparation -> heuristic intensity map ->
# meta-heuristic search -> training -> evaluation.
#
# Usage:
#   bash run_all.sh                      # full pipeline with PSO
#   bash run_all.sh ga_sa                # full pipeline with GA-SA
#   bash run_all.sh cs                   # full pipeline with Cuckoo Search
#
# Environment variables:
#   BASE_MODEL   (default: gpt2-medium)
#   DATA_DIR     (default: datasets/e2e_nlg_dataset)
#   SEED         (default: 42)
#   OUT_ROOT     (default: output)
#   SKIP_DATA=1  skip dataset preparation if it already exists

set -euo pipefail

ALGO="${1:-pso}"
BASE_MODEL="${BASE_MODEL:-gpt2-medium}"
DATA_DIR="${DATA_DIR:-datasets/e2e_nlg_dataset}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-output}"

echo "=============================================================="
echo " HeuFouFT pipeline"
echo " algorithm : ${ALGO}"
echo " base model: ${BASE_MODEL}"
echo " seed      : ${SEED}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Step 0: dataset preparation
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATA:-0}" != "1" && ! -d "${DATA_DIR}" ]]; then
    echo "[step 0] preparing the E2E dataset ..."
    python data/prepare_e2e.py --output_dir "${DATA_DIR}" --tokenizer "${BASE_MODEL}"
else
    echo "[step 0] dataset already present at ${DATA_DIR}, skipping"
fi

# ---------------------------------------------------------------------------
# Step 1: block probing -> heuristic intensity map (Section 3.1)
# ---------------------------------------------------------------------------
MAP_FILE="${OUT_ROOT}/probing/intensity_map.json"
if [[ ! -f "${MAP_FILE}" ]]; then
    echo "[step 1] building the heuristic intensity map ..."
    python src/block_probing.py \
        --base_model "${BASE_MODEL}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUT_ROOT}/probing" \
        --seed "${SEED}"
else
    echo "[step 1] intensity map found at ${MAP_FILE}, skipping"
fi

# ---------------------------------------------------------------------------
# Step 2: meta-heuristic search (Section 3.2)
# ---------------------------------------------------------------------------
INDICES_FILE="${OUT_ROOT}/search/${ALGO}/best_indices.npz"
if [[ ! -f "${INDICES_FILE}" ]]; then
    echo "[step 2] running ${ALGO} search ..."
    python src/run_search.py \
        --algorithm "${ALGO}" \
        --base_model "${BASE_MODEL}" \
        --data_dir "${DATA_DIR}" \
        --map_file "${MAP_FILE}" \
        --output_dir "${OUT_ROOT}/search/${ALGO}" \
        --n_frequency 625 \
        --seed "${SEED}"
else
    echo "[step 2] searched indices found at ${INDICES_FILE}, skipping"
fi

# ---------------------------------------------------------------------------
# Step 3: train with the searched coordinates (0.03M budget)
# ---------------------------------------------------------------------------
TRAIN_DIR="${OUT_ROOT}/train_${ALGO}"
if [[ ! -d "${TRAIN_DIR}/final_adapter" ]]; then
    echo "[step 3] training HeuFouFT (${ALGO}) ..."
    python src/train.py \
        --method fourierft --sampling indices \
        --indices_file "${INDICES_FILE}" \
        --base_model "${BASE_MODEL}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${TRAIN_DIR}" \
        --n_frequency 625 --scaling 16.0 \
        --learning_rate 2e-4 --seed "${SEED}"
else
    echo "[step 3] trained adapter found at ${TRAIN_DIR}/final_adapter, skipping"
fi

# ---------------------------------------------------------------------------
# Step 4: evaluate on the E2E test set
# ---------------------------------------------------------------------------
EVAL_DIR="${OUT_ROOT}/eval_${ALGO}"
echo "[step 4] evaluating ..."
python src/evaluate.py \
    --adapter_dir "${TRAIN_DIR}/final_adapter" \
    --base_model "${BASE_MODEL}" \
    --data_dir "${DATA_DIR}" \
    --split test \
    --output_dir "${EVAL_DIR}"

echo "=============================================================="
echo " Pipeline finished. Metrics: ${EVAL_DIR}/metrics.json"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Optional baselines (uncomment to reproduce the full Table 2):
# ---------------------------------------------------------------------------
# LoRA:
#   python src/train.py --method lora --learning_rate 3e-4 --seed ${SEED} \
#       --output_dir ${OUT_ROOT}/train_lora
#   python src/evaluate.py --adapter_dir ${OUT_ROOT}/train_lora/final_adapter ...
# Vanilla FourierFT (uniform):
#   python src/train.py --method fourierft --sampling uniform --seed ${SEED}
# Gaussian band-pass (frequency bias sweep for Table 1):
#   for FC in 0 100 200 300 400 500; do
#       python src/train.py --method fourierft --sampling bandpass --fc ${FC} ...
#   done
# Full fine-tuning:
#   python src/train.py --method full --learning_rate 5e-5 --seed ${SEED}
