#!/usr/bin/env python3
"""End-to-End Training Pipeline Orchestrator.

Implements OPS-01: Orchestrates the complete 3-stage curriculum learning pipeline
with automatic checkpoint handoff and validation.

Usage:
    # Dry-run (validate configs only):
    python scripts/run_pipeline.py --config config/pipeline.yaml --dry-run

    # Full pipeline:
    python scripts/run_pipeline.py --config config/pipeline.yaml

    # Resume from specific stage:
    python scripts/run_pipeline.py --config config/pipeline.yaml --start-from stage2
"""

import argparse
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_pipeline_config(config: Dict[str, Any]) -> None:
    """Validate pipeline configuration has all required fields."""
    required_fields = ["stage1", "stage2", "stage3", "output_base_dir"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in pipeline config: {field}")

    # Validate each stage has required fields
    for stage in ["stage1", "stage2", "stage3"]:
        stage_config = config[stage]
        if "config" not in stage_config:
            raise ValueError(f"Stage {stage} missing 'config' field")
        if "output_dir" not in stage_config:
            raise ValueError(f"Stage {stage} missing 'output_dir' field")


def run_stage1(config: Dict[str, Any], dry_run: bool = False) -> Path:
    """Execute Stage 1 training."""
    print("\n" + "=" * 80)
    print("STAGE 1: Pre-training on General Corpus")
    print("=" * 80 + "\n")

    stage_config = config["stage1"]
    output_dir = Path(stage_config["output_dir"])

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_stage1.py"),
        "--config",
        str(PROJECT_ROOT / stage_config["config"]),
        "--output-dir",
        str(output_dir),
    ]

    if dry_run:
        cmd.append("--dry-run")

    # Add optional arguments
    if "epochs" in stage_config:
        cmd.extend(["--epochs", str(stage_config["epochs"])])
    if "batch_size" in stage_config:
        cmd.extend(["--batch-size", str(stage_config["batch_size"])])

    print(f"Running command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage 1 training failed with exit code {result.returncode}"
        )

    print(f"\n✓ Stage 1 completed. Checkpoint saved to: {output_dir}")
    return output_dir


def run_stage2(
    config: Dict[str, Any], stage1_checkpoint: Path, dry_run: bool = False
) -> Path:
    """Execute Stage 2 domain adaptation."""
    print("\n" + "=" * 80)
    print("STAGE 2: Domain Adaptation (OARE)")
    print("=" * 80 + "\n")

    stage_config = config["stage2"]
    output_dir = Path(stage_config["output_dir"])

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_stage2.py"),
        "--checkpoint",
        str(stage1_checkpoint),
        "--config",
        str(PROJECT_ROOT / stage_config["config"]),
        "--output-dir",
        str(output_dir),
    ]

    if dry_run:
        cmd.append("--dry-run")

    # Add optional arguments
    if "epochs" in stage_config:
        cmd.extend(["--epochs", str(stage_config["epochs"])])
    if "batch_size" in stage_config:
        cmd.extend(["--batch-size", str(stage_config["batch_size"])])

    print(f"Running command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage 2 training failed with exit code {result.returncode}"
        )

    print(f"\n✓ Stage 2 completed. Checkpoint saved to: {output_dir}")
    return output_dir


def run_stage3(
    config: Dict[str, Any], stage2_checkpoint: Path, dry_run: bool = False
) -> Path:
    """Execute Stage 3 competition fine-tuning."""
    print("\n" + "=" * 80)
    print("STAGE 3: Competition Fine-Tuning")
    print("=" * 80 + "\n")

    stage_config = config["stage3"]
    output_dir = Path(stage_config["output_dir"])

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_stage3.py"),
        "--checkpoint",
        str(stage2_checkpoint),
        "--config",
        str(PROJECT_ROOT / stage_config["config"]),
        "--output-dir",
        str(output_dir),
    ]

    if dry_run:
        cmd.append("--dry-run")

    # Add optional arguments
    if "epochs" in stage_config:
        cmd.extend(["--epochs", str(stage_config["epochs"])])
    if "batch_size" in stage_config:
        cmd.extend(["--batch-size", str(stage_config["batch_size"])])
    if "early_stopping_patience" in stage_config:
        cmd.extend(
            ["--early-stopping-patience", str(stage_config["early_stopping_patience"])]
        )

    print(f"Running command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage 3 training failed with exit code {result.returncode}"
        )

    print(f"\n✓ Stage 3 completed. Checkpoint saved to: {output_dir}")
    return output_dir


def select_best_checkpoint(
    stage3_dir: Path, config: Dict[str, Any], dry_run: bool = False
) -> Path:
    """Select and export the best checkpoint from Stage 3."""
    print("\n" + "=" * 80)
    print("CHECKPOINT SELECTION: Finding Best Model")
    print("=" * 80 + "\n")

    export_dir = Path(config.get("final_model_dir", "models/submission_model"))

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_stage3.py"),
        "--checkpoints-dir",
        str(stage3_dir),
        "--export-dir",
        str(export_dir),
    ]

    if dry_run:
        cmd.append("--dry-run")

    if "val_data" in config["stage3"]:
        cmd.extend(["--val-data", str(PROJECT_ROOT / config["stage3"]["val_data"])])

    print(f"Running command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(
            f"Checkpoint selection failed with exit code {result.returncode}"
        )

    print(f"\n✓ Best checkpoint exported to: {export_dir}")
    return export_dir


def write_pipeline_summary(
    config: Dict[str, Any],
    stage1_dir: Path,
    stage2_dir: Path,
    stage3_dir: Path,
    final_model_dir: Path,
) -> None:
    """Write summary of pipeline execution."""
    output_dir = Path(config.get("output_base_dir", "outputs"))
    output_dir.mkdir(exist_ok=True, parents=True)

    summary = {
        "pipeline_config": str(config),
        "stage1_checkpoint": str(stage1_dir),
        "stage2_checkpoint": str(stage2_dir),
        "stage3_checkpoint": str(stage3_dir),
        "final_model": str(final_model_dir),
        "status": "completed",
    }

    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Pipeline summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Path to pipeline configuration YAML",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configurations without training",
    )

    parser.add_argument(
        "--start-from",
        choices=["stage1", "stage2", "stage3"],
        default="stage1",
        help="Start pipeline from specific stage (requires prior checkpoints)",
    )

    args = parser.parse_args()

    try:
        # Load and validate config
        print(f"Loading pipeline configuration from: {args.config}")
        config = load_pipeline_config(args.config)
        validate_pipeline_config(config)

        print("\n✓ Pipeline configuration validated")

        if args.dry_run:
            print("\nDRY RUN MODE: No actual training will be performed")

        # Execute pipeline stages
        stage1_dir = None
        stage2_dir = None
        stage3_dir = None

        if args.start_from == "stage1":
            stage1_dir = run_stage1(config, dry_run=args.dry_run)
            stage2_dir = run_stage2(config, stage1_dir, dry_run=args.dry_run)
            stage3_dir = run_stage3(config, stage2_dir, dry_run=args.dry_run)

        elif args.start_from == "stage2":
            stage1_dir = Path(config["stage1"]["output_dir"])
            if not stage1_dir.exists():
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_dir}")
            print(f"Using existing Stage 1 checkpoint: {stage1_dir}")
            stage2_dir = run_stage2(config, stage1_dir, dry_run=args.dry_run)
            stage3_dir = run_stage3(config, stage2_dir, dry_run=args.dry_run)

        elif args.start_from == "stage3":
            stage2_dir = Path(config["stage2"]["output_dir"])
            if not stage2_dir.exists():
                raise FileNotFoundError(f"Stage 2 checkpoint not found: {stage2_dir}")
            print(f"Using existing Stage 2 checkpoint: {stage2_dir}")
            stage3_dir = run_stage3(config, stage2_dir, dry_run=args.dry_run)

        # Select best checkpoint
        if not args.dry_run and stage3_dir:
            final_model_dir = select_best_checkpoint(
                stage3_dir, config, dry_run=args.dry_run
            )

            # Write summary
            write_pipeline_summary(
                config,
                stage1_dir or Path(config["stage1"]["output_dir"]),
                stage2_dir or Path(config["stage2"]["output_dir"]),
                stage3_dir,
                final_model_dir,
            )

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

        if not args.dry_run:
            print(f"\n🎉 Final model ready for inference at: {final_model_dir}")
            print(f"\nNext steps:")
            print(
                f"  1. Run inference: python scripts/run_inference.py --checkpoint {final_model_dir}"
            )
            print(f"  2. Validate submission: python scripts/validate_submission.py")

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
