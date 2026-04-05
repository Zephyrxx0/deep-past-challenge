#!/usr/bin/env python3
"""Run Reproduction Tool.

Implements OPS-02: Reproduces a training run from its manifest file,
validating environment and configuration consistency.

Usage:
    # Validate environment only:
    python scripts/reproduce_run.py --manifest models/stage3_final/run_manifest.json --validate-only

    # Full reproduction:
    python scripts/reproduce_run.py --manifest models/stage3_final/run_manifest.json
"""

import argparse
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load run manifest from JSON file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest


def validate_environment(manifest: Dict[str, Any], strict: bool = True) -> bool:
    """Validate current environment against manifest requirements."""
    print("\n" + "=" * 80)
    print("ENVIRONMENT VALIDATION")
    print("=" * 80 + "\n")

    issues = []
    warnings = []

    # Check Python version
    current_python = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    manifest_python = manifest.get("python_version", "unknown")

    if manifest_python != "unknown":
        print(f"Python version:")
        print(f"  Current:  {current_python}")
        print(f"  Manifest: {manifest_python}")

        if current_python != manifest_python:
            msg = f"Python version mismatch: {current_python} != {manifest_python}"
            if strict:
                issues.append(msg)
            else:
                warnings.append(msg)
        else:
            print("  ✓ Match\n")

    # Check platform
    current_platform = platform.system()
    manifest_platform = manifest.get("platform", "unknown")

    if manifest_platform != "unknown":
        print(f"Platform:")
        print(f"  Current:  {current_platform}")
        print(f"  Manifest: {manifest_platform}")

        if current_platform != manifest_platform:
            warnings.append(
                f"Platform difference: {current_platform} != {manifest_platform}"
            )
        else:
            print("  ✓ Match\n")

    # Check git hash (if available)
    if "git_hash" in manifest:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            current_hash = result.stdout.strip()
            manifest_hash = manifest["git_hash"]

            print(f"Git commit:")
            print(f"  Current:  {current_hash[:8]}")
            print(f"  Manifest: {manifest_hash[:8]}")

            if current_hash != manifest_hash:
                warnings.append(f"Git commit mismatch: code may have changed since run")
            else:
                print("  ✓ Match\n")
        except:
            warnings.append("Could not verify git commit hash")

    # Check config file exists
    if "config_path" in manifest:
        config_path = Path(manifest["config_path"])
        print(f"Configuration file:")
        print(f"  Path: {config_path}")

        if not config_path.exists():
            issues.append(f"Config file not found: {config_path}")
        else:
            print("  ✓ Exists\n")

    # Check data files exist
    if "train_csv" in manifest:
        train_csv = Path(manifest["train_csv"])
        print(f"Training data:")
        print(f"  Path: {train_csv}")

        if not train_csv.exists():
            issues.append(f"Training data not found: {train_csv}")
        else:
            print("  ✓ Exists\n")

    # Print results
    if issues:
        print("\n❌ VALIDATION FAILED")
        print("\nCritical Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    if warnings:
        print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nReproduction may not be exact due to environment differences.")
    else:
        print("✓ VALIDATION PASSED")
        print("\nEnvironment matches manifest requirements.")

    return True


def reproduce_run(manifest: Dict[str, Any], validate_only: bool = False) -> int:
    """Reproduce a training run from manifest."""
    print("\n" + "=" * 80)
    print("RUN REPRODUCTION")
    print("=" * 80 + "\n")

    # Validate environment first
    if not validate_environment(manifest, strict=False):
        print("\n❌ Cannot reproduce run: environment validation failed")
        return 1

    if validate_only:
        print("\n✓ Validation complete (--validate-only mode)")
        return 0

    # Determine stage and training script
    stage = manifest.get("stage", 1)
    stage_map = {
        1: "train_stage1.py",
        2: "train_stage2.py",
        3: "train_stage3.py",
        "stage1": "train_stage1.py",
        "stage2": "train_stage2.py",
        "stage3": "train_stage3.py",
    }

    script_name = stage_map.get(stage)
    if not script_name:
        print(f"❌ Unknown stage in manifest: {stage}")
        return 1

    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        print(f"❌ Training script not found: {script_path}")
        return 1

    # Build command from manifest
    cmd = [sys.executable, str(script_path)]

    # Add config
    if "config_path" in manifest:
        cmd.extend(["--config", manifest["config_path"]])

    # Add output directory (with _reproduced suffix)
    if "output_dir" in manifest:
        output_dir = Path(manifest["output_dir"]).parent / (
            Path(manifest["output_dir"]).name + "_reproduced"
        )
        cmd.extend(["--output-dir", str(output_dir)])

    # Add checkpoint (for stage 2/3)
    if "checkpoint_path" in manifest and stage in [2, 3, "stage2", "stage3"]:
        cmd.extend(["--checkpoint", manifest["checkpoint_path"]])

    # Add other parameters from manifest
    if "epochs" in manifest:
        cmd.extend(["--epochs", str(manifest["epochs"])])
    if "batch_size" in manifest:
        cmd.extend(["--batch-size", str(manifest["batch_size"])])
    if "learning_rate" in manifest:
        cmd.extend(["--learning-rate", str(manifest["learning_rate"])])

    print(f"Reproducing Stage {stage} training run...")
    print(f"Command: {' '.join(cmd)}\n")
    print("Press Ctrl+C to cancel...\n")

    # Execute training
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print("\n✓ Run reproduced successfully")
        if "output_dir" in manifest:
            print(f"  Output saved to: {output_dir}")
        return 0
    else:
        print(f"\n❌ Reproduction failed with exit code {result.returncode}")
        return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce a training run from its manifest file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to run_manifest.json file"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment, don't reproduce run",
    )

    args = parser.parse_args()

    try:
        # Load manifest
        print(f"Loading manifest from: {args.manifest}")
        manifest = load_manifest(args.manifest)

        print(f"\nManifest Details:")
        print(f"  Stage: {manifest.get('stage', 'unknown')}")
        print(f"  Model: {manifest.get('model_id', 'unknown')}")
        print(f"  Timestamp: {manifest.get('timestamp', 'unknown')}")

        # Reproduce run
        return reproduce_run(manifest, validate_only=args.validate_only)

    except Exception as e:
        print(f"\n❌ Reproduction failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
