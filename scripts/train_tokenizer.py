import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer with project special tokens."
    )
    parser.add_argument(
        "--input-file", required=True, help="Path to normalized training text file."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory override for tokenizer artifacts.",
    )
    parser.add_argument(
        "--config",
        default="config/tokenizer/tokenizer.yaml",
        help="Tokenizer YAML config path.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return payload


def train_tokenizer(
    input_file: Path, output_dir: Path, config: dict
) -> tuple[Path, Path, dict]:
    if not input_file.exists():
        raise FileNotFoundError(f"input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / str(config["model_prefix"]))

    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=model_prefix,
        vocab_size=int(config["vocab_size"]),
        character_coverage=float(config["character_coverage"]),
        model_type=str(config["model_type"]),
        normalization_rule_name=str(config["normalization_rule_name"]),
        user_defined_symbols=list(config["special_tokens"]),
        hard_vocab_limit=False,
        minloglevel=2,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    model_path = output_dir / f"{config['model_prefix']}.model"
    vocab_path = output_dir / f"{config['model_prefix']}.vocab"
    manifest = {
        "model_prefix": str(config["model_prefix"]),
        "vocab_size": int(config["vocab_size"]),
        "special_tokens": list(config["special_tokens"]),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_path": str(model_path.resolve()),
        "vocab_path": str(vocab_path.resolve()),
    }
    return model_path, vocab_path, manifest


def main() -> int:
    args = parse_args()
    try:
        config = load_config(Path(args.config))
        output_dir = Path(
            args.output_dir or config.get("default_output_dir", "models/tokenizer")
        )
        model_path, vocab_path, manifest = train_tokenizer(
            Path(args.input_file), output_dir, config
        )

        manifest_path = output_dir / "tokenizer_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        print(f"model_path: {model_path.resolve()}")
        print(f"vocab_path: {vocab_path.resolve()}")
        print(f"manifest_path: {manifest_path.resolve()}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
