import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve tokenizer artifacts from manifest as JSON."
    )
    parser.add_argument(
        "--manifest",
        default="models/tokenizer/tokenizer_manifest.json",
        help="Path to tokenizer manifest JSON.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        default=True,
        help="Print JSON output (default true).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        model_path = Path(manifest.get("model_path", ""))
        vocab_path = Path(manifest.get("vocab_path", ""))
        special_tokens = manifest.get("special_tokens")

        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"vocab not found: {vocab_path}")
        if not isinstance(special_tokens, list):
            raise ValueError("special_tokens missing or invalid in manifest")

        payload = {
            "model_path": str(model_path.resolve()),
            "vocab_path": str(vocab_path.resolve()),
            "special_tokens": special_tokens,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
