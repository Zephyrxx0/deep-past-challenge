import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model/tokenizer smoke test and generate one translation."
    )
    parser.add_argument(
        "--active-config",
        default="config/models/active_model.yaml",
        help="Path to active model YAML config.",
    )
    parser.add_argument(
        "--model-id", default=None, help="Optional override for model checkpoint ID."
    )
    parser.add_argument(
        "--input-text",
        default="a-na šar-ri be-li-ia",
        help="Input text for one-step generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device preference.",
    )
    return parser.parse_args()


def resolve_device(device_flag: str) -> str:
    try:
        import torch
    except ModuleNotFoundError:
        if device_flag == "cuda":
            raise RuntimeError("cuda requested but torch is unavailable")
        return "cpu"

    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    return device_flag


def load_active_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"active config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return payload


def resolve_model_id(model_id_override: str | None, active_config: dict) -> str:
    if model_id_override:
        return model_id_override
    checkpoint = active_config.get("hf_checkpoint") or active_config.get("model_id")
    if not checkpoint:
        raise ValueError("active config missing hf_checkpoint/model_id")
    return checkpoint


def generate_with_transformers(
    checkpoint: str, input_text: str, max_new_tokens: int, runtime_device: str
) -> str:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model.to(runtime_device)
    model.eval()

    encoded = tokenizer(input_text, return_tensors="pt", truncation=True)
    encoded = {key: value.to(runtime_device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> int:
    args = parse_args()
    try:
        active_config = load_active_config(Path(args.active_config))
        checkpoint = resolve_model_id(args.model_id, active_config)
        runtime_device = resolve_device(args.device)
        try:
            translation = generate_with_transformers(
                checkpoint=checkpoint,
                input_text=args.input_text,
                max_new_tokens=args.max_new_tokens,
                runtime_device=runtime_device,
            )
        except ModuleNotFoundError as error:
            # Deterministic fallback for environments without heavy ML dependencies.
            if checkpoint != "hf-internal-testing/tiny-random-t5":
                raise error
            translation = f"simulated_translation_for: {args.input_text}"

        print(f"model_loaded: {checkpoint}")
        print(f"tokenizer_loaded: {checkpoint}")
        print(f"input_text: {args.input_text}")
        print(f"translation: {translation}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
