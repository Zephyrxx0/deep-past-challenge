import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an approved translation model and persist active config."
    )
    parser.add_argument(
        "--model-id", required=True, help="Approved model identifier to activate."
    )
    parser.add_argument(
        "--catalog",
        default="config/models/catalog.yaml",
        help="Path to approved model catalog YAML.",
    )
    parser.add_argument(
        "--output",
        default="config/models/active_model.yaml",
        help="Path to output active model YAML.",
    )
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"catalog file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    models = payload.get("approved_models", [])
    if not isinstance(models, list):
        raise ValueError("invalid catalog format: approved_models must be a list")

    return models


def select_model(model_id: str, catalog: list[dict]) -> dict:
    for entry in catalog:
        if entry.get("model_id") == model_id:
            return {
                "model_id": entry["model_id"],
                "hf_checkpoint": entry["hf_checkpoint"],
                "selected_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
            }
    raise ValueError(f"unsupported model: {model_id}")


def write_active_model(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as temp_file:
        yaml.safe_dump(payload, temp_file, sort_keys=False)
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog)
    output_path = Path(args.output)

    try:
        catalog = load_catalog(catalog_path)
        active_payload = select_model(args.model_id, catalog)
        write_active_model(output_path, active_payload)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}")
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": active_payload["model_id"],
                "output": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
