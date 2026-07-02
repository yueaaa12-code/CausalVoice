#!/bin/bash
# CausalVoice Training Script
# Phase 1: Generate synthetic causal data
# Phase 2: Train with causal regularization

set -e

# ============ Configuration ============
CAUSAL_DATA="data/synthetic_causal"
REAL_DATA="filelists/train_real.txt"
LOG_DIR="logs/causal_voice"
CONFIG="configs/causal_vc.json"
BASELINE_CKPT=""  # Set to pretrained O²-VC checkpoint if available

# ============ Phase 1: Generate Interventions ============
echo "=== Phase 1: Generating Causal Intervention Data ==="
python causal/generate_interventions.py \
    --output_dir ${CAUSAL_DATA}/train \
    --num_groups 5000 \
    --n_speaker_interventions 3 \
    --n_noise_interventions 2 \
    --n_speed_interventions 1 \
    --seed 42

python causal/generate_interventions.py \
    --output_dir ${CAUSAL_DATA}/test \
    --num_groups 500 \
    --n_speaker_interventions 3 \
    --n_noise_interventions 2 \
    --n_speed_interventions 1 \
    --seed 123

echo "=== Phase 1 Complete ==="

# ============ Phase 2: Train CausalVoice ============
echo "=== Phase 2: Training CausalVoice ==="

EXTRA_ARGS=""
if [ -n "$BASELINE_CKPT" ]; then
    EXTRA_ARGS="--weight_path ${BASELINE_CKPT} --causal_start_epoch 0"
fi

python train_causal.py \
    -c ${CONFIG} \
    -m ${LOG_DIR} \
    --causal_data_path ${CAUSAL_DATA}/train \
    --lambda_contrastive 1000.0 \
    --lambda_ranking 1000.0 \
    --reg_type both \
    --causal_batch_size 8 \
    ${EXTRA_ARGS}

echo "=== Phase 2 Complete ==="

# ============ Phase 3: Evaluate ============
echo "=== Phase 3: Evaluating Causal Awareness ==="
BEST_CKPT=$(ls -t ${LOG_DIR}/G_*.pth | head -1)

python evaluate_causal.py \
    --checkpoint ${BEST_CKPT} \
    --causal_test_path ${CAUSAL_DATA}/test \
    --config ${CONFIG}

echo "=== All Done ==="
