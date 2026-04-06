# Deep Past GUI

A modern, user-friendly interface for training and using the Akkadian-English translation model.

## Features

- **📊 Dataset Management**: Upload and validate training datasets for all 3 stages
- **🎯 Training Configuration**: Configure hyperparameters and run the training pipeline
- **🔮 Translation**: Load trained models and translate Akkadian text to English
- **📈 Progress Tracking**: Real-time training logs and progress visualization

## Quick Start

### Windows
```batch
run_gui.bat
```

### Linux/macOS
```bash
chmod +x run_gui.sh
./run_gui.sh
```

### Manual Launch
```bash
pip install gradio>=4.0.0
python gui/app.py
```

## Usage

### 1. Upload Datasets

Navigate to the **Datasets** tab and upload your training data:

| Stage | File | Description |
|-------|------|-------------|
| Stage 1 | `stage1_train.csv` | General Akkadian corpus |
| Stage 2 | `stage2_train.csv` | OARE domain data |
| Stage 3 | `stage3_train.csv` | Competition training data |
| Stage 3 | `stage3_val.csv` | Validation data (optional) |

**Required columns:**
- `transliteration` - Akkadian text in Latin transliteration
- `translation` - English translation

### 2. Configure Training

Navigate to the **Training** tab to configure:

- **Epochs** for each stage
- **Batch size** (reduce if OOM)
- **Learning rate**
- **FP16 / Gradient checkpointing** for memory optimization

### 3. Run Training

Click **Start Training** to begin the 3-stage curriculum learning pipeline.

### 4. Translate

After training, navigate to the **Translate** tab:

1. Load your trained model
2. Enter Akkadian text
3. Click **Translate**

## Command Line Options

```bash
python gui/app.py --port 7861        # Custom port
python gui/app.py --share            # Create public link
python gui/app.py --debug            # Enable debug mode
```

## Screenshots

*Coming soon*

## Requirements

- Python 3.8+
- Gradio 4.0+
- PyTorch
- Transformers

See `requirements.txt` for full list.
