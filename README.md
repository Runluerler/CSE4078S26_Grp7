# CSE4078 Spring 2026 - Group 7 Project

This repository contains the code and resources for fine-tuning and evaluating small Large Language Models (LLMs) on Turkish medical question-answering, as part of the CSE4078 term project.

## Project Overview
- **Base Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Dataset:** `alibayram/doktorsitesi` (filtered for mandatory medical fields, resulting in ~39k train / 1.5k test instances)
- **Goal:** Evaluate the performance of the base model, fine-tune it using QLoRA/LoRA, and analyze the improvements in Turkish medical domain context.

## Repository Structure
- `data/`: Contains the train and test CSV files.
- `evaluation/`: Scripts to run inference and compute evaluation metrics (`evaluate_model.py`, `compare_results.py`).
- `models/`: Directory where the fine-tuned LoRA adapters are saved.
- `results/`: Contains the CSV outputs from the base and fine-tuned models, along with error analysis files.

## How to Reproduce

### 1. Setup Environment
Ensure you have the required libraries installed:
```bash
pip install torch transformers peft pandas evaluate tqdm rouge_score
```

### 2. Fine-Tuning the Model (SFT)
*(To be completed by the training lead)*
```bash
# Example command for running the training script
python scripts/training/train.py --data data/train_dataset.csv --output_dir models/qwen_lora
```

### 3. Running Evaluation (Inference)
Use the `evaluate_model.py` script to generate answers for the 1500-question test dataset.

**For the Base Model:**
```bash
python evaluation/evaluate_model.py \
    --data data/test_dataset.csv \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --output results/qwen_base_outputs.csv
```

**For the Fine-Tuned Model:**
```bash
python evaluation/evaluate_model.py \
    --data data/test_dataset.csv \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path models/qwen_lora \
    --output results/qwen_lora_outputs.csv
```
*(Note: You can use `--limit 20` to test the script quickly before running on all 1500 rows).*

### 4. Comparing Results & Metrics
To compare the two models and calculate metrics (like ROUGE, answer lengths, empty answer counts):
```bash
python evaluation/compare_results.py \
    --base results/qwen_base_outputs.csv \
    --finetuned results/qwen_lora_outputs.csv \
    --output_dir results/
```
This will generate `comparison_summary.csv` and a combined `before_after_outputs.csv`.

### 5. Error Analysis
To extract a randomized sample across different medical categories for manual human review:
```bash
python evaluation/create_error_sample.py \
    --before_after results/before_after_outputs.csv \
    --output results/error_analysis_sample.csv \
    --n_per_class 5
```
You can then open `error_analysis_sample.csv` in Excel to manually label improvements or hallucinations.