import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print stage-specific training config as JSON."
    )
    parser.add_argument(
        "--stage", required=True, choices=["stage1", "stage2", "stage3"]
    )
    parser.add_argument(
        "--config-dir",
        default="config/training",
        help="Directory containing stage YAML files.",
    )
    return parser.parse_args()


def load_stage_config(config_dir: Path, stage: str) -> dict:
    config_path = config_dir / f"{stage}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"stage config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir)
    try:
        payload = load_stage_config(config_dir, args.stage)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}")
        return 2

    print(json.dumps({"stage": args.stage, "config": payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
