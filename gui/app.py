#!/usr/bin/env python3
"""Deep Past GUI - Akkadian-English Translation Training Interface.

A modern, user-friendly interface for training and using the Akkadian-English
translation model with curriculum learning.

Usage:
    python gui/app.py

    # Or with custom port:
    python gui/app.py --port 7861
"""

import os
import sys
import json
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import gradio as gr
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

# CSS for custom styling
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.main-header {
    text-align: center;
    margin-bottom: 1rem;
}

.stage-card {
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.status-ready {
    color: #10b981;
    font-weight: bold;
}

.status-missing {
    color: #ef4444;
    font-weight: bold;
}

.status-running {
    color: #f59e0b;
    font-weight: bold;
}

.dataset-info {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

footer {
    display: none !important;
}
"""

# Theme configuration
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_label_text_weight="500",
)


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================


class DatasetManager:
    """Manages dataset loading, validation, and storage."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, Optional[Path]] = {
            "stage1": None,
            "stage2": None,
            "stage3_train": None,
            "stage3_val": None,
        }
        self._load_existing_datasets()

    def _load_existing_datasets(self):
        """Check for existing datasets in data directory."""
        mappings = {
            "stage1": "stage1_train.csv",
            "stage2": "stage2_train.csv",
            "stage3_train": "stage3_train.csv",
            "stage3_val": "stage3_val.csv",
        }

        for key, filename in mappings.items():
            path = self.data_dir / filename
            if path.exists():
                self.datasets[key] = path

    def validate_dataset(
        self, file_path: Path, stage: str
    ) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Validate a dataset file for a specific stage.

        Returns:
            Tuple of (is_valid, message, dataframe_preview)
        """
        try:
            df = pd.read_csv(file_path)

            # Check for required columns
            if stage in ["stage1", "stage3_train", "stage3_val"]:
                required_cols = ["transliteration", "translation"]
            else:  # stage2
                required_cols = ["first_word_spelling", "translation"]

            # Also accept normalized column names
            alt_cols = ["transliteration_normalized", "translation_normalized"]

            has_required = all(col in df.columns for col in required_cols)
            has_alt = all(col in df.columns for col in alt_cols)

            if not (has_required or has_alt):
                return (
                    False,
                    f"Missing required columns. Expected: {required_cols}",
                    None,
                )

            # Check for null values
            if has_required:
                null_count = df[required_cols].isnull().sum().sum()
            else:
                null_count = df[alt_cols].isnull().sum().sum()

            rows = len(df)
            info = f"✓ Valid dataset: {rows:,} rows"
            if null_count > 0:
                info += f" ({null_count} null values will be dropped)"

            return True, info, df.head(5)

        except Exception as e:
            return False, f"Error reading file: {str(e)}", None

    def save_dataset(self, file_obj, stage: str) -> Tuple[bool, str]:
        """Save an uploaded dataset file."""
        if file_obj is None:
            return False, "No file provided"

        # Map stage to filename
        filename_map = {
            "stage1": "stage1_train.csv",
            "stage2": "stage2_train.csv",
            "stage3_train": "stage3_train.csv",
            "stage3_val": "stage3_val.csv",
        }

        target_path = self.data_dir / filename_map[stage]

        try:
            # Copy file to data directory
            shutil.copy(file_obj, target_path)
            self.datasets[stage] = target_path
            return True, f"✓ Dataset saved to {target_path}"
        except Exception as e:
            return False, f"Error saving file: {str(e)}"

    def get_dataset_info(self, stage: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        path = self.datasets.get(stage)

        if path is None or not path.exists():
            return {
                "exists": False,
                "path": None,
                "rows": 0,
                "columns": [],
            }

        try:
            df = pd.read_csv(path)
            return {
                "exists": True,
                "path": str(path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        except Exception:
            return {
                "exists": False,
                "path": str(path),
                "rows": 0,
                "columns": [],
            }

    def get_all_status(self) -> str:
        """Get formatted status of all datasets."""
        lines = ["### Dataset Status\n"]

        stage_names = {
            "stage1": "Stage 1 (General Corpus)",
            "stage2": "Stage 2 (OARE Domain)",
            "stage3_train": "Stage 3 Training",
            "stage3_val": "Stage 3 Validation",
        }

        all_ready = True
        for stage, name in stage_names.items():
            info = self.get_dataset_info(stage)
            if info["exists"]:
                lines.append(f"- **{name}**: ✅ {info['rows']:,} rows")
            else:
                lines.append(f"- **{name}**: ❌ Missing")
                if stage != "stage3_val":  # val is optional
                    all_ready = False

        lines.append("")
        if all_ready:
            lines.append("**Status: Ready for training** ✅")
        else:
            lines.append("**Status: Upload missing datasets** ⚠️")

        return "\n".join(lines)


# ============================================================================
# TRAINING MANAGER
# ============================================================================


class TrainingManager:
    """Manages training execution and progress tracking."""

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.is_training = False
        self.current_stage = None
        self.progress = 0.0
        self.logs = []
        self.output_dir = DEFAULT_OUTPUT_DIR

    def start_training(
        self,
        stage1_epochs: int,
        stage2_epochs: int,
        stage3_epochs: int,
        batch_size: int,
        learning_rate: float,
        use_fp16: bool,
        progress_callback=None,
    ) -> str:
        """Start the training pipeline."""
        if self.is_training:
            return "Training already in progress!"

        # Validate datasets
        for stage in ["stage1", "stage2", "stage3_train"]:
            info = self.dataset_manager.get_dataset_info(stage)
            if not info["exists"]:
                return f"❌ Missing dataset: {stage}"

        self.is_training = True
        self.logs = []

        try:
            # Import training modules
            from scripts.train_stage1 import main as train_stage1
            from scripts.train_stage2 import main as train_stage2
            from scripts.train_stage3 import main as train_stage3

            # This would run the actual training
            # For now, we'll simulate it
            self._log("Starting 3-stage curriculum learning pipeline...")
            self._log(f"Configuration:")
            self._log(f"  - Stage 1: {stage1_epochs} epochs")
            self._log(f"  - Stage 2: {stage2_epochs} epochs")
            self._log(f"  - Stage 3: {stage3_epochs} epochs")
            self._log(f"  - Batch size: {batch_size}")
            self._log(f"  - Learning rate: {learning_rate}")
            self._log(f"  - FP16: {use_fp16}")

            return "\n".join(self.logs)

        except Exception as e:
            self.is_training = False
            return f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False

    def _log(self, message: str):
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")


# ============================================================================
# INFERENCE MANAGER
# ============================================================================


class InferenceManager:
    """Manages model inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None

    def load_model(self, model_path: str) -> str:
        """Load a trained model for inference."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            path = Path(model_path)
            if not path.exists():
                return f"❌ Model not found: {model_path}"

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
            self.model_path = path

            return f"✅ Model loaded from {path}"

        except Exception as e:
            return f"❌ Failed to load model: {str(e)}"

    def translate(self, text: str, num_beams: int = 4, max_length: int = 256) -> str:
        """Translate Akkadian text to English."""
        if self.model is None:
            return "❌ Please load a model first"

        try:
            import torch

            inputs = self.tokenizer(text, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation

        except Exception as e:
            return f"❌ Translation failed: {str(e)}"


# ============================================================================
# GUI APPLICATION
# ============================================================================


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    dataset_manager = DatasetManager()
    training_manager = TrainingManager(dataset_manager)
    inference_manager = InferenceManager()

    with gr.Blocks(title="Deep Past - Akkadian Translator") as app:
        # Header
        gr.Markdown(
            """
            # 🏛️ Deep Past: Akkadian-English Translation
            
            A curriculum learning pipeline for translating ancient Akkadian texts to English.
            
            ---
            """,
            elem_classes=["main-header"],
        )

        with gr.Tabs() as tabs:
            # ================================================================
            # TAB 1: Dataset Management
            # ================================================================
            with gr.TabItem("📊 Datasets", id="datasets"):
                gr.Markdown("### Upload and Manage Training Datasets")

                with gr.Row():
                    with gr.Column(scale=2):
                        # Dataset status
                        dataset_status = gr.Markdown(
                            dataset_manager.get_all_status(),
                            elem_classes=["dataset-info"],
                        )

                        refresh_btn = gr.Button(
                            "🔄 Refresh Status", variant="secondary"
                        )

                    with gr.Column(scale=3):
                        gr.Markdown("### Upload Datasets")

                        with gr.Accordion("Stage 1: General Corpus", open=True):
                            stage1_file = gr.File(
                                label="Upload stage1_train.csv",
                                file_types=[".csv"],
                                type="filepath",
                            )
                            stage1_info = gr.Markdown("")
                            stage1_preview = gr.Dataframe(
                                label="Preview",
                                visible=False,
                            )

                        with gr.Accordion("Stage 2: OARE Domain", open=True):
                            stage2_file = gr.File(
                                label="Upload stage2_train.csv",
                                file_types=[".csv"],
                                type="filepath",
                            )
                            stage2_info = gr.Markdown("")
                            stage2_preview = gr.Dataframe(
                                label="Preview",
                                visible=False,
                            )

                        with gr.Accordion("Stage 3: Competition Data", open=True):
                            stage3_train_file = gr.File(
                                label="Upload stage3_train.csv",
                                file_types=[".csv"],
                                type="filepath",
                            )
                            stage3_train_info = gr.Markdown("")

                            stage3_val_file = gr.File(
                                label="Upload stage3_val.csv (optional)",
                                file_types=[".csv"],
                                type="filepath",
                            )
                            stage3_val_info = gr.Markdown("")

                # Event handlers for dataset uploads
                def handle_stage1_upload(file):
                    if file is None:
                        return (
                            "",
                            gr.update(visible=False),
                            dataset_manager.get_all_status(),
                        )
                    valid, msg, preview = dataset_manager.validate_dataset(
                        Path(file), "stage1"
                    )
                    if valid:
                        dataset_manager.save_dataset(file, "stage1")
                    return (
                        msg,
                        gr.update(value=preview, visible=valid),
                        dataset_manager.get_all_status(),
                    )

                def handle_stage2_upload(file):
                    if file is None:
                        return (
                            "",
                            gr.update(visible=False),
                            dataset_manager.get_all_status(),
                        )
                    valid, msg, preview = dataset_manager.validate_dataset(
                        Path(file), "stage2"
                    )
                    if valid:
                        dataset_manager.save_dataset(file, "stage2")
                    return (
                        msg,
                        gr.update(value=preview, visible=valid),
                        dataset_manager.get_all_status(),
                    )

                def handle_stage3_train_upload(file):
                    if file is None:
                        return "", dataset_manager.get_all_status()
                    valid, msg, _ = dataset_manager.validate_dataset(
                        Path(file), "stage3_train"
                    )
                    if valid:
                        dataset_manager.save_dataset(file, "stage3_train")
                    return msg, dataset_manager.get_all_status()

                def handle_stage3_val_upload(file):
                    if file is None:
                        return "", dataset_manager.get_all_status()
                    valid, msg, _ = dataset_manager.validate_dataset(
                        Path(file), "stage3_val"
                    )
                    if valid:
                        dataset_manager.save_dataset(file, "stage3_val")
                    return msg, dataset_manager.get_all_status()

                stage1_file.change(
                    handle_stage1_upload,
                    inputs=[stage1_file],
                    outputs=[stage1_info, stage1_preview, dataset_status],
                )

                stage2_file.change(
                    handle_stage2_upload,
                    inputs=[stage2_file],
                    outputs=[stage2_info, stage2_preview, dataset_status],
                )

                stage3_train_file.change(
                    handle_stage3_train_upload,
                    inputs=[stage3_train_file],
                    outputs=[stage3_train_info, dataset_status],
                )

                stage3_val_file.change(
                    handle_stage3_val_upload,
                    inputs=[stage3_val_file],
                    outputs=[stage3_val_info, dataset_status],
                )

                refresh_btn.click(
                    lambda: dataset_manager.get_all_status(),
                    outputs=[dataset_status],
                )

            # ================================================================
            # TAB 2: Training Configuration
            # ================================================================
            with gr.TabItem("🎯 Training", id="training"):
                gr.Markdown("### Configure and Run Training Pipeline")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Stage Configuration")

                        with gr.Group():
                            gr.Markdown("**Stage 1: General Corpus**")
                            stage1_epochs = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Epochs",
                            )

                        with gr.Group():
                            gr.Markdown("**Stage 2: Domain Adaptation**")
                            stage2_epochs = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Epochs",
                            )

                        with gr.Group():
                            gr.Markdown("**Stage 3: Fine-Tuning**")
                            stage3_epochs = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=10,
                                step=1,
                                label="Epochs",
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("#### General Settings")

                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="Batch Size",
                            info="Reduce if you run out of memory",
                        )

                        learning_rate = gr.Number(
                            value=1e-4,
                            label="Learning Rate (Stage 1)",
                            info="Stages 2 & 3 use progressively lower rates",
                        )

                        use_fp16 = gr.Checkbox(
                            value=True,
                            label="Use FP16 (Half Precision)",
                            info="Reduces memory usage, recommended for most GPUs",
                        )

                        gradient_checkpointing = gr.Checkbox(
                            value=True,
                            label="Gradient Checkpointing",
                            info="Trades compute for memory, useful for large models",
                        )

                gr.Markdown("---")

                with gr.Row():
                    start_training_btn = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        scale=2,
                    )
                    stop_training_btn = gr.Button(
                        "⏹️ Stop",
                        variant="stop",
                        scale=1,
                    )

                training_progress = gr.Progress()
                training_output = gr.Textbox(
                    label="Training Log",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                )

                def start_training(s1_epochs, s2_epochs, s3_epochs, batch, lr, fp16):
                    return training_manager.start_training(
                        stage1_epochs=s1_epochs,
                        stage2_epochs=s2_epochs,
                        stage3_epochs=s3_epochs,
                        batch_size=batch,
                        learning_rate=lr,
                        use_fp16=fp16,
                    )

                start_training_btn.click(
                    start_training,
                    inputs=[
                        stage1_epochs,
                        stage2_epochs,
                        stage3_epochs,
                        batch_size,
                        learning_rate,
                        use_fp16,
                    ],
                    outputs=[training_output],
                )

            # ================================================================
            # TAB 3: Inference
            # ================================================================
            with gr.TabItem("🔮 Translate", id="inference"):
                gr.Markdown("### Translate Akkadian Text")

                with gr.Row():
                    with gr.Column(scale=1):
                        model_path = gr.Textbox(
                            label="Model Path",
                            value=str(DEFAULT_OUTPUT_DIR / "submission_model"),
                            info="Path to trained model checkpoint",
                        )
                        load_model_btn = gr.Button("📥 Load Model", variant="secondary")
                        model_status = gr.Markdown("*No model loaded*")

                    with gr.Column(scale=1):
                        num_beams = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Beam Search Width",
                            info="Higher = better quality but slower",
                        )
                        max_length = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=256,
                            step=32,
                            label="Max Output Length",
                        )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Akkadian Text (Transliteration)",
                            placeholder="e.g., a-na LUGAL be-li2-ia",
                            lines=4,
                        )
                        translate_btn = gr.Button("🔄 Translate", variant="primary")

                    with gr.Column():
                        output_text = gr.Textbox(
                            label="English Translation",
                            lines=4,
                            interactive=False,
                        )

                # Example inputs
                gr.Markdown("#### Examples")
                gr.Examples(
                    examples=[
                        ["a-na LUGAL be-li2-ia"],
                        ["um-ma {disz}a-bi-{d}UTU-ma"],
                        ["a-na pa-ni {m}a-bi-ra-am"],
                    ],
                    inputs=[input_text],
                )

                def load_model_handler(path):
                    result = inference_manager.load_model(path)
                    return result

                def translate_handler(text, beams, max_len):
                    return inference_manager.translate(text, beams, max_len)

                load_model_btn.click(
                    load_model_handler,
                    inputs=[model_path],
                    outputs=[model_status],
                )

                translate_btn.click(
                    translate_handler,
                    inputs=[input_text, num_beams, max_length],
                    outputs=[output_text],
                )

            # ================================================================
            # TAB 4: About
            # ================================================================
            with gr.TabItem("ℹ️ About", id="about"):
                gr.Markdown(
                    """
                    ## About Deep Past
                    
                    This application implements a **3-stage curriculum learning pipeline** for 
                    training neural machine translation models to translate ancient Akkadian 
                    cuneiform texts into English.
                    
                    ### Pipeline Stages
                    
                    1. **Stage 1: Pre-training** - Train on a large general Akkadian corpus
                    2. **Stage 2: Domain Adaptation** - Fine-tune on OARE domain data
                    3. **Stage 3: Competition Fine-Tuning** - Final fine-tuning with early stopping
                    
                    ### Model Architecture
                    
                    - **Base Model**: mBART-large-50 (multilingual BART)
                    - **Languages**: Arabic (ar_AR) as proxy for Akkadian → English (en_XX)
                    
                    ### Dataset Requirements
                    
                    Each stage dataset should be a CSV file with these columns:
                    
                    | Column | Description |
                    |--------|-------------|
                    | `transliteration` | Akkadian text in Latin transliteration |
                    | `translation` | English translation |
                    
                    Stage 2 alternatively accepts:
                    - `first_word_spelling` (for Akkadian)
                    - `translation` (for English)
                    
                    ### Resources
                    
                    - [GitHub Repository](https://github.com/Zephyrxx0/deep-past-challenge)
                    - [Kaggle Competition](https://www.kaggle.com/competitions/deep-past)
                    
                    ---
                    
                    *Built with ❤️ using Gradio and Transformers*
                    """
                )

        # Footer
        gr.Markdown(
            """
            ---
            <center>
            <small>Deep Past v1.0.0 | © 2024</small>
            </center>
            """,
        )

    return app


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Launch the GUI application."""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Past GUI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    app = create_app()

    print("\n" + "=" * 60)
    print("🏛️  Deep Past - Akkadian Translation GUI")
    print("=" * 60)
    print(f"\n🌐 Starting server on port {args.port}...")

    app.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
