import argparse
import json
import unicodedata
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic tokenizer encode/decode roundtrip checks."
    )
    parser.add_argument(
        "--samples-file",
        required=True,
        help="Path to newline-delimited transliteration samples.",
    )
    parser.add_argument(
        "--manifest",
        default="models/tokenizer/tokenizer_manifest.json",
        help="Path to tokenizer manifest JSON.",
    )
    parser.add_argument(
        "--strict-special-token-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require source special tokens to appear as exact pieces in encoding.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload


def load_samples(samples_file: Path) -> list[str]:
    if not samples_file.exists():
        raise FileNotFoundError(f"samples file not found: {samples_file}")
    lines = [
        line.strip() for line in samples_file.read_text(encoding="utf-8").splitlines()
    ]
    return [line for line in lines if line]


def check_roundtrip(
    samples: list[str],
    processor: spm.SentencePieceProcessor,
    special_tokens: list[str],
    strict_special_token_check: bool,
) -> tuple[int, int, list[str]]:
    passed = 0
    failed = 0
    errors: list[str] = []

    for index, sample in enumerate(samples, start=1):
        ids = processor.encode(sample, out_type=int)
        pieces = processor.encode(sample, out_type=str)
        decoded = processor.decode(ids)

        if normalize_space(sample) != normalize_space(decoded):
            failed += 1
            errors.append(f"line {index}: roundtrip mismatch")
            continue

        if strict_special_token_check:
            for token in special_tokens:
                if token in sample and token not in pieces:
                    failed += 1
                    errors.append(
                        f"line {index}: special token not preserved as piece ({token})"
                    )
                    break
            else:
                passed += 1
        else:
            passed += 1

    return passed, failed, errors


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(Path(args.manifest))
        model_path = Path(manifest.get("model_path", ""))
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        special_tokens = manifest.get("special_tokens")
        if not isinstance(special_tokens, list):
            raise ValueError("special_tokens missing or invalid in manifest")

        samples = load_samples(Path(args.samples_file))
        processor = spm.SentencePieceProcessor(model_file=str(model_path))

        passed, failed, errors = check_roundtrip(
            samples=samples,
            processor=processor,
            special_tokens=special_tokens,
            strict_special_token_check=args.strict_special_token_check,
        )

        checked = len(samples)
        print(f"samples_checked: {checked}")
        print(f"roundtrip_passed: {passed}")
        print(f"roundtrip_failed: {failed}")

        if failed > 0:
            detail = "; ".join(errors)
            print(f"error: roundtrip failures detected ({detail})")
            return 1
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
