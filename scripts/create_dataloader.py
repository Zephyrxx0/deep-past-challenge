import argparse
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    Dataset = object

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    AutoTokenizer = None

from scripts.data_loader import load_stage_data


SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SOURCE_LENGTH = 256
DEFAULT_MAX_TARGET_LENGTH = 256

# mBART language codes
# For Akkadian->English, we use a close proxy: Arabic (ar_AR) for Akkadian (Semitic family)
# and English (en_XX) for target. This provides reasonable subword tokenization.
DEFAULT_SRC_LANG = "ar_AR"  # Closest available proxy for Akkadian
DEFAULT_TGT_LANG = "en_XX"  # English


class TranslationDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.df = df
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        source = row["transliteration_normalized"]
        target = row["translation_normalized"]
        return {"source": source, "target": target}


def collate_fn(batch, tokenizer, max_source_length: int, max_target_length: int):
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]

    # Tokenize sources and targets together using text_target parameter
    # This is the modern approach (replaces deprecated as_target_tokenizer)
    model_inputs = tokenizer(
        sources,
        text_target=targets,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Replace padding token id with -100 for labels (ignored in loss calculation)
    labels = model_inputs["labels"]
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels,
    }


def create_dataloader(
    stage: int,
    model_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split: str = "train",
    max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
    max_target_length: int = DEFAULT_MAX_TARGET_LENGTH,
    seed: int = SEED,
    src_lang: str = DEFAULT_SRC_LANG,
    tgt_lang: str = DEFAULT_TGT_LANG,
) -> DataLoader:
    if torch is None:
        raise ModuleNotFoundError("torch is required to create DataLoader")
    if AutoTokenizer is None:
        raise ModuleNotFoundError("transformers is required to create DataLoader")

    df = load_stage_data(stage, split, seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set source and target languages for mBART models
    # This is required to properly configure the tokenizer for translation tasks
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang

    dataset = TranslationDataset(
        df,
        tokenizer,
        max_source_length,
        max_target_length,
    )

    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    return loader


def main():
    parser = argparse.ArgumentParser(
        description="Create PyTorch DataLoader with seq2seq collation"
    )
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--max-source-length", type=int, default=DEFAULT_MAX_SOURCE_LENGTH
    )
    parser.add_argument(
        "--max-target-length", type=int, default=DEFAULT_MAX_TARGET_LENGTH
    )
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    try:
        loader = create_dataloader(
            args.stage,
            args.model_id,
            args.batch_size,
            args.split,
            args.max_source_length,
            args.max_target_length,
            args.seed,
        )

        print(
            f"DataLoader created for stage {args.stage} ({args.split}): "
            f"{len(loader)} batches of size {args.batch_size}"
        )

        batch = next(iter(loader))
        print(
            f"Batch keys: {list(batch.keys())}, "
            f"input_ids shape: {batch['input_ids'].shape}, "
            f"labels shape: {batch['labels'].shape}"
        )

        sys.exit(0)

    except ModuleNotFoundError as e:
        if str(e).startswith("torch is required"):
            print(
                "Warning: torch is not installed; skipping DataLoader creation.",
                file=sys.stderr,
            )
            print(
                "Install dependencies with: pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(0)
        if str(e).startswith("transformers is required"):
            print(
                "Warning: transformers is not installed; skipping DataLoader creation.",
                file=sys.stderr,
            )
            print(
                "Install dependencies with: pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(0)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
