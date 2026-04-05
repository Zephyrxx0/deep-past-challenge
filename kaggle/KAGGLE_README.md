# Running Deep Past on Kaggle

This guide explains how to run the Akkadian-English translation model training and inference on Kaggle.

## Quick Start

### Option 1: Use the Notebooks Directly

1. **Upload your data as a Kaggle dataset**
   - Go to Kaggle → Datasets → New Dataset
   - Upload the `data/` folder contents
   - Name it something like `deep-past-akkadian`

2. **Create a new notebook**
   - Go to Kaggle → Code → New Notebook
   - Add your dataset: Click "Add data" → Select your uploaded dataset

3. **Copy notebook contents**
   - Copy the contents of `kaggle_training.ipynb` into your Kaggle notebook
   - Update `KAGGLE_DATASET_NAME` to match your dataset name
   - Enable GPU: Settings → Accelerator → GPU (P100 or T4)
   - Run all cells

4. **Run inference**
   - After training, create a new notebook
   - Copy contents of `kaggle_inference.ipynb`
   - Add your trained model (from step 3) as input data
   - Run to generate `submission.csv`

### Option 2: Upload Full Repository

1. **Create a Kaggle dataset with the full repo**
   ```
   deep-past-akkadian/
   ├── data/
   │   ├── stage1_train.csv
   │   ├── stage2_train.csv
   │   ├── stage3_train.csv
   │   ├── stage3_val.csv
   │   └── competition/
   │       └── test.csv
   ├── config/
   │   └── ...
   ├── kaggle/
   │   ├── kaggle_training.ipynb
   │   └── kaggle_inference.ipynb
   └── scripts/
       └── ...
   ```

2. **Import the notebook**
   - In your Kaggle notebook, import from your dataset

---

## Directory Structure on Kaggle

When running on Kaggle:

```
/kaggle/
├── input/                    # READ-ONLY: Your datasets
│   └── deep-past-akkadian/   # Your uploaded dataset
│       ├── data/
│       │   ├── stage1_train.csv
│       │   ├── stage2_train.csv
│       │   └── ...
│       └── kaggle/
│           └── notebooks...
│
└── working/                  # WRITABLE: Output directory
    ├── models/
    │   ├── stage1_final/
    │   ├── stage2_final/
    │   ├── stage3_final/
    │   └── submission_model/
    ├── submission.csv
    └── ...
```

**Important:** Kaggle input directories are read-only. All outputs (models, checkpoints, submissions) are written to `/kaggle/working/`.

---

## Configuration

### Update Dataset Name

In the notebooks, update this variable to match your Kaggle dataset name:

```python
KAGGLE_DATASET_NAME = "deep-past-akkadian"  # Change to your dataset name
```

### Adjust Batch Size

Based on GPU memory (Kaggle P100 has 16GB):

```python
# For P100/T4 (16GB)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 16

# If you get OOM errors
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Still effective batch = 16
```

### Reduce Training Time

For faster iteration (not recommended for final training):

```python
STAGE_CONFIG = {
    1: {"epochs": 2, ...},  # Reduced from 5
    2: {"epochs": 2, ...},  # Reduced from 3
    3: {"epochs": 5, ...},  # Reduced from 10
}
```

---

## GPU Selection

Kaggle offers these accelerators:
- **P100** (16GB) - Best for this model
- **T4** (16GB) - Good alternative
- **TPU v3-8** - Requires code changes (not supported in these notebooks)

To enable GPU:
1. Click the accelerator dropdown in notebook settings
2. Select "GPU P100" or "GPU T4"

---

## Saving and Loading Models

### Save model for later use

After training, your model is saved to `/kaggle/working/models/submission_model/`.

To use it in another notebook:
1. Go to your training notebook
2. Click "Output" tab on the right
3. Save as dataset: "Save Version" → "Save Output"
4. In your inference notebook, add this dataset as input

### Load a pre-trained model

If you have a pre-trained model:

```python
# In inference notebook
MODEL_DIR = Path("/kaggle/input/your-model-dataset/models/submission_model")
```

---

## Troubleshooting

### "FileNotFoundError: Data not found"

1. Check your dataset was added to the notebook
2. Verify the dataset name matches `KAGGLE_DATASET_NAME`
3. Check the data directory structure matches expected paths

### "CUDA out of memory"

1. Reduce `BATCH_SIZE` (try 2 or 1)
2. Increase `GRADIENT_ACCUMULATION_STEPS` proportionally
3. Reduce `MAX_SOURCE_LENGTH` and `MAX_TARGET_LENGTH`

### Training is slow

1. Ensure GPU is enabled (check "GPU" appears in kernel info)
2. Consider reducing number of epochs for initial testing
3. Make sure batch size isn't too small (too many gradient accumulation steps)

### Kernel dies during training

1. Check GPU memory usage
2. Reduce batch size
3. Save checkpoints more frequently
4. Consider using Kaggle's "quick save" feature

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│                    PREPARATION                          │
├─────────────────────────────────────────────────────────┤
│ 1. Upload data/ folder as Kaggle dataset                │
│ 2. Create new Kaggle notebook                           │
│ 3. Add dataset to notebook                              │
│ 4. Enable GPU accelerator                               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    TRAINING                             │
├─────────────────────────────────────────────────────────┤
│ 1. Run kaggle_training.ipynb                            │
│ 2. Stage 1: Pre-training (~1-2 hours)                   │
│ 3. Stage 2: Domain adaptation (~30-60 min)              │
│ 4. Stage 3: Fine-tuning (~1-2 hours)                    │
│ 5. Save trained model as output dataset                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    INFERENCE                            │
├─────────────────────────────────────────────────────────┤
│ 1. Create new notebook                                  │
│ 2. Add trained model dataset                            │
│ 3. Add competition test data                            │
│ 4. Run kaggle_inference.ipynb                           │
│ 5. Download submission.csv                              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    SUBMIT                               │
├─────────────────────────────────────────────────────────┤
│ Submit submission.csv to competition                    │
└─────────────────────────────────────────────────────────┘
```

---

## Files in this Directory

| File | Description |
|------|-------------|
| `kaggle_training.ipynb` | Complete 3-stage training notebook |
| `kaggle_inference.ipynb` | Inference and submission generation |
| `kaggle_config.py` | Configuration module for Kaggle paths |
| `requirements.txt` | Python dependencies (Kaggle-compatible) |
| `KAGGLE_README.md` | This file |

---

## Estimated Training Time

On Kaggle P100 GPU with default settings:

| Stage | Epochs | Approx. Time |
|-------|--------|--------------|
| Stage 1 | 5 | 1-2 hours |
| Stage 2 | 3 | 30-60 min |
| Stage 3 | 10 (early stopping) | 1-2 hours |
| **Total** | - | **3-5 hours** |

Note: Kaggle notebooks have a 12-hour runtime limit. The training should complete well within this limit.
