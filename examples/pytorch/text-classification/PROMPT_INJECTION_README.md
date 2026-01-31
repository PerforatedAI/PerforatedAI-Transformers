# Prompt Injection Detection Training

This modified version of `run_classification_perforatedai_meta.py` is configured to train models for prompt injection detection using three combined datasets.

## Features

### Combined Datasets
The script combines three prompt injection/jailbreak detection datasets:
1. **jackhhao/jailbreak-classification** (~1.3K samples) - jailbreak vs benign prompts
2. **deepset/prompt-injections** (~662 samples) - injection vs legitimate prompts  
3. **qualifire/prompt-injections-benchmark** (~5K samples) - jailbreak vs benign prompts

Total: ~7K samples for training and validation

All datasets are automatically:
- Downloaded from HuggingFace
- Standardized to binary labels (0=benign/legitimate, 1=jailbreak/injection)
- Combined into a single training dataset
- Split 90/10 into train/validation sets

### Model Selection
Choose between two DeBERTa-v3 models via command line arguments:
- **microsoft/deberta-v3-base** (default) - 184M parameters
- **microsoft/deberta-v3-xsmall** - 22M parameters (faster training)

### Recommended Training Settings

The script automatically applies recommended hyperparameters for each model:

**DeBERTa-v3-base:**
- Learning rate: 1e-5
- Batch size: 8
- Epochs: 3
- Better accuracy, slower training

**DeBERTa-v3-xsmall:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 5
- Faster training, good for prototyping

## Usage

### Basic Training with DeBERTa-v3-base (default)
```bash
python run_classification_perforatedai_meta.py \
    --use_combined_prompt_injection_datasets \
    --do_train \
    --do_eval \
    --output_dir ./results/deberta-base-prompt-injection
```

### Training with DeBERTa-v3-xsmall
```bash
python run_classification_perforatedai_meta.py \
    --use_combined_prompt_injection_datasets \
    --use_deberta_xsmall \
    --do_train \
    --do_eval \
    --output_dir ./results/deberta-xsmall-prompt-injection
```

### Override Recommended Settings
You can override the recommended settings with custom values:
```bash
python run_classification_perforatedai_meta.py \
    --use_combined_prompt_injection_datasets \
    --use_deberta_xsmall \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 4 \
    --do_train \
    --do_eval \
    --output_dir ./results/custom-settings
```

### Additional Useful Arguments
```bash
--max_seq_length 256              # Maximum sequence length (default: 128)
--save_strategy epoch             # Save checkpoint every epoch
--evaluation_strategy epoch       # Evaluate every epoch
--logging_steps 100               # Log every 100 steps
--warmup_ratio 0.1                # Warmup 10% of steps
--weight_decay 0.01               # L2 regularization
--load_best_model_at_end          # Load best checkpoint at end
--metric_for_best_model eval_accuracy  # Metric to determine best model
--fp16                            # Use mixed precision (faster on GPU)
--gradient_accumulation_steps 2   # Accumulate gradients for larger effective batch
```

## Quick Start Script

A ready-to-use example script is provided:
```bash
chmod +x train_prompt_injection.sh
./train_prompt_injection.sh
```

Edit the script to choose between the two models or customize settings.

## New Command Line Arguments

### `--use_combined_prompt_injection_datasets`
- **Type:** flag
- **Default:** False
- **Description:** Enable training on the combined three prompt injection datasets

### `--use_deberta_xsmall`
- **Type:** flag
- **Default:** False
- **Description:** Use microsoft/deberta-v3-xsmall instead of microsoft/deberta-v3-base

## Label Mapping

The combined dataset uses binary classification:
- **Label 0:** Benign/Legitimate prompts (safe)
- **Label 1:** Jailbreak/Injection prompts (harmful)

## Dataset Access Note

The **qualifire/prompt-injections-benchmark** dataset requires accepting their terms of use on HuggingFace. You may need to:
1. Log in to HuggingFace: `huggingface-cli login`
2. Accept the dataset terms at: https://huggingface.co/datasets/qualifire/prompt-injections-benchmark

## Output

The trained model will be saved to the specified `--output_dir` along with:
- Model checkpoints
- Training metrics (train_results.json)
- Evaluation metrics (eval_results.json)
- Tokenizer configuration

## Example Results Structure
```
./results/deberta-base-prompt-injection/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── train_results.json
├── eval_results.json
└── checkpoint-XXX/
```

## Compatibility

This script maintains compatibility with all original features of `run_classification_perforatedai_meta.py`. You can still:
- Use custom datasets via `--dataset_name`
- Load from CSV/JSON files
- Use other HuggingFace models
- Apply all standard training arguments
