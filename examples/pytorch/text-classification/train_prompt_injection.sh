#!/bin/bash

# Example script to train on combined prompt injection datasets

# Train with DeBERTa-v3-base (default)
# This will use recommended settings: lr=1e-5, batch_size=8, epochs=3
CUDA_VISIBLE_DEVICES=0 python run_classification_perforatedai_meta.py \
    --model-name-or-path microsoft/deberta-v3-base \
    --use-combined-prompt-injection-datasets \
    --do-train \
    --do-eval \
    --output-dir ./results/deberta-v3-base-prompt-injection \
    --per-device-eval-batch-size 8 \
    --save-strategy epoch \
    --logging-steps 100 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01

# Train with DeBERTa-v3-xsmall
# This will use recommended settings: lr=2e-5, batch_size=16, epochs=5
# CUDA_VISIBLE_DEVICES=0 python run_classification_perforatedai_meta.py \
#     --model-name-or-path microsoft/deberta-v3-xsmall \
#     --use-combined-prompt-injection-datasets \
#     --use-deberta-xsmall \
#     --do-train \
#     --do-eval \
#     --output-dir ./results/deberta-v3-xsmall-prompt-injection \
#     --per-device-eval-batch-size 16 \
#     --save-strategy epoch \
#     --logging-steps 100 \
#     --warmup-ratio 0.1 \
#     --weight-decay 0.01

# CUDA_VISIBLE_DEVICES=0 python run_classification_perforatedai_meta.py \
#     --model_name_or_path microsoft/deberta-v3-xsmall with custom values:
# python run_classification_perforatedai_meta.py \
#     --use_combined_prompt_injection_datasets \
#     --use_deberta_xsmall \
#     --learning_rate 3e-5 \
#     --per_device_train_batch_size 32 \
#     --num_train_epochs 4 \
#     --do_train \
#     --do_eval \
#     --output_dir ./results/deberta-v3-xsmall-custom
