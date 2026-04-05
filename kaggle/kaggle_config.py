"""
Kaggle Environment Configuration for Deep Past Akkadian-English Translation

This module provides configuration and path management for running the
training pipeline on Kaggle. It handles:
- Automatic detection of Kaggle vs local environment
- Path mapping for Kaggle input datasets
- Output directory configuration for Kaggle
- GPU/TPU detection and configuration
"""

import os
import sys
from pathlib import Path
from typing import Optional


def is_kaggle_environment() -> bool:
    """Detect if running in Kaggle environment."""
    return os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_colab_environment() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab

        return True
    except ImportError:
        return False


# Environment detection
RUNNING_ON_KAGGLE = is_kaggle_environment()
RUNNING_ON_COLAB = is_colab_environment()
RUNNING_LOCALLY = not RUNNING_ON_KAGGLE and not RUNNING_ON_COLAB


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

if RUNNING_ON_KAGGLE:
    # Kaggle paths - datasets are mounted read-only at /kaggle/input/<dataset-name>
    # User needs to add the competition dataset to their notebook

    # Competition data input (user must add "deep-past-challenge" dataset)
    KAGGLE_INPUT_BASE = Path("/kaggle/input")

    # Common Kaggle dataset names (user may need to adjust)
    # Option 1: If user uploads the full repo as a dataset
    REPO_DATASET_NAME = "deep-past-akkadian"  # Adjust this to your dataset name

    # Option 2: If using competition dataset directly
    COMPETITION_DATASET_NAME = "deep-past"  # Adjust to competition slug

    # Working directory for outputs (writable)
    KAGGLE_WORKING = Path("/kaggle/working")

    # Data paths - try multiple locations
    def find_data_path() -> Path:
        """Find the data directory in Kaggle environment."""
        possible_paths = [
            KAGGLE_INPUT_BASE / REPO_DATASET_NAME / "data",
            KAGGLE_INPUT_BASE / COMPETITION_DATASET_NAME,
            KAGGLE_INPUT_BASE / "deep-past-challenge" / "data",
            KAGGLE_INPUT_BASE / "akkadian-translation",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # List available datasets for debugging
        if KAGGLE_INPUT_BASE.exists():
            available = list(KAGGLE_INPUT_BASE.iterdir())
            print(f"[WARN] Could not find data. Available datasets: {available}")

        # Default fallback
        return KAGGLE_INPUT_BASE / REPO_DATASET_NAME / "data"

    DATA_DIR = find_data_path()

    # Output directories (all go to /kaggle/working)
    OUTPUT_BASE = KAGGLE_WORKING / "outputs"
    MODELS_DIR = KAGGLE_WORKING / "models"
    CHECKPOINTS_DIR = MODELS_DIR

    # Config path
    def find_config_path() -> Path:
        """Find the config directory in Kaggle environment."""
        possible_paths = [
            KAGGLE_INPUT_BASE / REPO_DATASET_NAME / "config",
            KAGGLE_WORKING / "config",
            Path("config"),
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return Path("config")

    CONFIG_DIR = find_config_path()

elif RUNNING_ON_COLAB:
    # Google Colab paths
    COLAB_BASE = Path("/content")

    # Assume repo is cloned to /content/deep-past-challenge
    REPO_DIR = COLAB_BASE / "deep-past-challenge"

    DATA_DIR = REPO_DIR / "data"
    CONFIG_DIR = REPO_DIR / "config"
    OUTPUT_BASE = REPO_DIR / "outputs"
    MODELS_DIR = REPO_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR

else:
    # Local development paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    DATA_DIR = PROJECT_ROOT / "data"
    CONFIG_DIR = PROJECT_ROOT / "config"
    OUTPUT_BASE = PROJECT_ROOT / "outputs"
    MODELS_DIR = PROJECT_ROOT / "models"
    CHECKPOINTS_DIR = MODELS_DIR


# ============================================================================
# STAGE-SPECIFIC DATA PATHS
# ============================================================================


def get_stage_data_paths(stage: int) -> dict:
    """Get data file paths for a training stage.

    Args:
        stage: Training stage (1, 2, or 3)

    Returns:
        Dictionary with 'train' and optionally 'val' paths
    """
    paths = {
        1: {
            "train": DATA_DIR / "stage1_train.csv",
        },
        2: {
            "train": DATA_DIR / "stage2_train.csv",
        },
        3: {
            "train": DATA_DIR / "stage3_train.csv",
            "val": DATA_DIR / "stage3_val.csv",
        },
    }
    return paths.get(stage, {})


def get_competition_test_path() -> Path:
    """Get path to competition test CSV."""
    possible_paths = [
        DATA_DIR / "competition" / "test.csv",
        DATA_DIR / "test.csv",
        DATA_DIR / "competition_test.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Default
    return DATA_DIR / "competition" / "test.csv"


def get_glossary_path() -> Path:
    """Get path to glossary JSON file."""
    return DATA_DIR / "glossary.json"


# ============================================================================
# MODEL & CHECKPOINT PATHS
# ============================================================================


def get_stage_output_dir(stage: int) -> Path:
    """Get output directory for a training stage."""
    return MODELS_DIR / f"stage{stage}_final"


def get_submission_model_dir() -> Path:
    """Get directory for final submission model."""
    return MODELS_DIR / "submission_model"


def get_inference_output_dir() -> Path:
    """Get directory for inference outputs."""
    return OUTPUT_BASE / "inference"


# ============================================================================
# CONFIG PATHS
# ============================================================================


def get_stage_config_path(stage: int) -> Path:
    """Get config YAML path for a training stage."""
    return CONFIG_DIR / "training" / f"stage{stage}.yaml"


def get_pipeline_config_path() -> Path:
    """Get pipeline config YAML path."""
    return CONFIG_DIR / "pipeline.yaml"


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================


def get_device():
    """Get the best available compute device."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device
    else:
        print("[WARN] No GPU available, using CPU (training will be slow)")
        return torch.device("cpu")


def get_optimal_batch_size(model_size: str = "mbart-large") -> int:
    """Get optimal batch size based on available GPU memory.

    Args:
        model_size: Model size identifier

    Returns:
        Recommended batch size
    """
    import torch

    if not torch.cuda.is_available():
        return 2  # Small batch for CPU

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Kaggle typically provides:
    # - P100: 16 GB
    # - T4: 16 GB
    # - TPU v3-8: distributed

    if model_size == "mbart-large":
        if gpu_memory_gb >= 16:
            return 4
        elif gpu_memory_gb >= 12:
            return 2
        else:
            return 1
    else:
        return 4


def get_gradient_accumulation_steps(target_effective_batch: int = 16) -> int:
    """Calculate gradient accumulation steps for target effective batch size."""
    actual_batch = get_optimal_batch_size()
    return max(1, target_effective_batch // actual_batch)


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================


def setup_kaggle_environment():
    """Perform Kaggle-specific environment setup."""
    if not RUNNING_ON_KAGGLE:
        return

    # Create output directories
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Add scripts to path if available
    scripts_path = Path("/kaggle/input") / "deep-past-akkadian" / "scripts"
    if scripts_path.exists() and str(scripts_path.parent) not in sys.path:
        sys.path.insert(0, str(scripts_path.parent))

    print(f"[INFO] Kaggle environment setup complete")
    print(f"[INFO] Data directory: {DATA_DIR}")
    print(f"[INFO] Output directory: {OUTPUT_BASE}")
    print(f"[INFO] Models directory: {MODELS_DIR}")


def print_environment_info():
    """Print current environment configuration."""
    print("=" * 60)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 60)

    env_type = (
        "Kaggle" if RUNNING_ON_KAGGLE else "Colab" if RUNNING_ON_COLAB else "Local"
    )
    print(f"Environment: {env_type}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Config Directory: {CONFIG_DIR}")
    print(f"Output Directory: {OUTPUT_BASE}")
    print(f"Models Directory: {MODELS_DIR}")

    # Check data availability
    print("\nData Files:")
    for stage in [1, 2, 3]:
        paths = get_stage_data_paths(stage)
        for split, path in paths.items():
            status = "OK" if path.exists() else "MISSING"
            print(f"  Stage {stage} {split}: {status} ({path})")

    # Check competition test
    test_path = get_competition_test_path()
    status = "OK" if test_path.exists() else "MISSING"
    print(f"  Competition test: {status} ({test_path})")

    print("=" * 60)


# Auto-setup on import in Kaggle
if RUNNING_ON_KAGGLE:
    setup_kaggle_environment()
