# Deep Past Akkadian-English Translation: Run Instructions

This guide provides step-by-step instructions on how to train, evaluate, and generate competition submissions for the Deep Past Akkadian-English translation system. All commands below are formatted for Windows PowerShell.

---

## 🔧 Prerequisites & Setup

Before running anything, ensure your Python environment is set up and activated.

```powershell
# 1. Create a virtual environment (if you haven't already)
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

*(Ensure your dataset files are present in the `data/` directory: `stage1_train.csv`, `stage2_train.csv`, `stage3_train.csv`, `stage3_val.csv`, and `competition/test.csv`.)*

---

## 🚀 Method 1: The Automated Pipeline (Recommended)

The easiest way to execute the entire 3-stage curriculum learning pipeline is using the automated orchestrator. This script handles checkpoint handoffs, validation, and training continuity automatically.

```powershell
# First, validate your pipeline configuration (dry-run mode)
python scripts/run_pipeline.py --config config/pipeline.yaml --dry-run

# Once validation passes, run the full training pipeline
python scripts/run_pipeline.py --config config/pipeline.yaml
```

**Advanced Options:**
```powershell
# Resume from a specific stage (if you already have earlier checkpoints)
python scripts/run_pipeline.py --config config/pipeline.yaml --start-from stage2
python scripts/run_pipeline.py --config config/pipeline.yaml --start-from stage3
```

Once the pipeline completes, proceed directly to **Step 4: Inference & Submission** below.

---

## 🛠️ Method 2: Manual Execution (Stage-by-Stage)

If you need fine-grained control, want to tune hyperparameters per stage, or resume a specific part of the process, use the manual stage-by-stage execution.

### Step 1: Pre-training (Stage 1)
Train the foundational Akkadian-English model on the general corpus.

```powershell
# 1a. Train the model
python scripts/train_stage1.py --config config/training/stage1.yaml --output-dir models/stage1_final

# 1b. (Optional) Evaluate Stage 1 performance
python scripts/evaluate_stage1.py --checkpoint models/stage1_final/epoch_1 --val-data data/stage1_val.csv
```

### Step 2: Domain Adaptation (Stage 2)
Adapt the foundational model to the specific OARE domain. This uses a lower learning rate.

```powershell
# 2a. Train the model using the Stage 1 checkpoint as the starting point
python scripts/train_stage2.py --checkpoint models/stage1_final --config config/training/stage2.yaml --output-dir models/stage2_final

# 2b. Check for catastrophic forgetting against the Stage 1 baseline
python scripts/forgetting_detection.py --stage1-checkpoint models/stage1_final --stage2-checkpoint models/stage2_final/epoch_1 --val-data data/stage1_val.csv
```

### Step 3: Competition Fine-Tuning (Stage 3)
Fine-tune for the competition using genre-conditioned data and early stopping.

```powershell
# 3a. Train the model using the Stage 2 checkpoint
python scripts/train_stage3.py --checkpoint models/stage2_final --config config/training/stage3.yaml --output-dir models/stage3_final

# 3b. Evaluate and select the BEST checkpoint from Stage 3
# This automatically evaluates all epochs and exports the single best model to the specified directory
python scripts/evaluate_stage3.py --checkpoints-dir models/stage3_final --val-data data/stage3_val.csv --export-dir models/submission_model
```

---

## 🎯 Step 4: Inference & Submission

Once you have your final model (either from the automated pipeline or manual Step 3), you can generate predictions for the competition test set.

### 1. Run Batched Inference
Generate translations for the held-out competition dataset using your best model.

```powershell
python scripts/run_inference.py --checkpoint models/submission_model --test-csv data/competition/test.csv --output outputs/submission_raw.csv --batch-size 16
```

### 2. (Optional) Apply Glossary Post-Processing
Improve proper-name accuracy using the predefined glossary.

```powershell
python scripts/apply_glossary.py --input outputs/submission_raw.csv --glossary data/glossary.json --output outputs/submission_final.csv --report outputs/glossary_report.json
```

### 3. Validate Submission Format
Ensure your final CSV complies strictly with the competition schema (no missing IDs, no duplicates, correct headers).

```powershell
# If using the glossary output:
python scripts/validate_submission.py --submission outputs/submission_final.csv --test-data data/competition/test.csv

# If skipping glossary:
python scripts/validate_submission.py --submission outputs/submission_raw.csv --test-data data/competition/test.csv
```

If validation returns `PASS`, your file is ready to be uploaded to the competition platform! 🎉

---

## 📊 Evaluation Metrics (Anytime)
If you want to compute BLEU, chrF, and get a genre-specific breakdown for any checkpoint at any time:

```powershell
python scripts/evaluate.py --checkpoint models/submission_model --val-data data/stage3_val.csv --genre-breakdown --output outputs/eval_results.json
```
