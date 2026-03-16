# Data Summary — Deep Past Challenge

This document explains which files/datasets are used, the train/val/test splits and staging, and how differently encoded transliterations (glyphs, subscripts, diacritics) are normalized and handled for English translation.

---

## 1) Datasets used (source files and columns)

- Competition / OARE (folder: `data/competition/`)
  - `train.csv` — primary competition training set. Columns used: `transliteration`, `translation`, `oare_id` (for tablet/document id).
  - `test.csv` — held-out competition test set (no target translations). Kept untouched for inference.
  - `sample_submission.csv` — submission format.
  - `published_texts.csv` — used to extract `genre_label` (merged into the competition train).
  - `Sentences_Oare_FirstWord_LinNum.csv` — used as `sentences_oare_df` for Stage 2 (contains `text_uuid`, `translation`, `first_word_spelling`, `line_number`, `side`).
  - `OA_Lexicon_eBL.csv` — lexicon used to construct `glossary.json` (columns: `form`, `norm`, `type` — used `PN`, `GN`, `MN`).
  - `eBL_Dictionary.csv` — optional dictionary resource (not required by pipeline but available).

- Akkademia (folder: `data/akkademia/`)
  - Parallel files: `train.ak`, `train.en`, `valid.ak`, `valid.en`, `test.ak`, `test.en`.
  - Each `.ak` is Akkadian transliteration, each `.en` is the aligned English translation. Loaded into `akkademia_data['train|valid|test']` DataFrames with columns `transliteration`, `translation`.

- ORACC Kaggle (folder: `data/oracc_kaggle/`)
  - `train.csv` — contains `akkadian` and `english` columns; used as an external large corpus (loaded to `oracc_df`).

Notes:
- All datasets are read from local `data/` subfolders and then normalized to `*_normalized` columns.

---

## 2) What files/splits are used for training, validation, and testing

Processed outputs (saved to `data/` by the notebook):
- `data/stage1_train.csv` — Stage 1 training set (Akkademia `train` + `valid` after deduplication + ORACC deduped). Purpose: general Akkadian foundation training.
- `data/stage2_train.csv` — Stage 2 training set (Sentences_Oare filtered). Purpose: domain adaptation to OARE style.
- `data/stage3_train.csv` — Stage 3 fine-tuning set (competition `train` after prefixing genres), doc-order 90% portion.
- `data/stage3_val.csv` — Stage 3 validation set (competition `train` doc-order 10% portion) used for monitoring fine-tuning.
- `data/glossary.json` — proper-name glossary extracted from `OA_Lexicon_eBL.csv`.

Split strategy and rationale:
- Competition data (`train.csv`) is the target domain and is split in **document order**: first 90% → `stage3_train.csv`, final 10% → `stage3_val.csv`.
  - Rationale: avoids placing fragments from the same document/tablet into both train and val (reduces leakage).
- External corpora (Akkademia, ORACC) are used to build Stage 1 (no train/val split inside the pipeline file — they are combined and cleaned to form the large foundation set).
- `Sentences_Oare_FirstWord_LinNum.csv` is filtered (tablet-level exclusion if tablet appears in competition train) and used exclusively for Stage 2.
- `test.csv` from `data/competition/` is kept as the final held-out test set; the pipeline does not alter it.

If you need exact numeric counts, run the notebook cells that print `len(...)` for these DataFrames (the pipeline computes counts into `summary` variables and prints them).

---

## 3) How differently encoded transliterations and glyphs are reconciled for translation

Problem statement:
- The same Akkadian content can be represented differently across sources:
  - subscript Unicode digits vs ASCII digits (e.g. `a₁` vs `a1`)
  - precomposed glyph characters with diacritics (`š`, `ṣ`, `ā`) vs decomposed sequences (base + combining marks)
  - editorial markers and lacunae (`[broken]`, `[...]`) and scribal markers (`!`, `?`, `#`)
- If left unnormalized, each variant becomes a distinct token and explodes vocabulary, hurting model learning.

Normalization strategy (what the notebook does):
1. Unicode NFC normalization (`unicodedata.normalize('NFC', ...)`)
   - Converts decomposed characters (base + combining marks) into precomposed glyphs when possible, ensuring consistent byte/char representation across sources.
2. Subscript digits → ASCII digits mapping
   - Maps Unicode subscripts (`₁₂₃...`) to ASCII digits (`1,2,3`) to reduce token heterogeneity.
3. Replace bracketed lacunae/damage with a single `<gap>` token
   - E.g. `[broken]` or `[...something...]` → `<gap>` so the model learns a uniform representation for missing text.
4. Strip scribal/editorial markers (`!`, `?`, `#`) from transliteration
   - These indicate uncertainty or editorial notes but are not part of the linguistic content the model should translate.
5. Normalize whitespace (collapse runs of space into single spaces; strip edges).
6. Normalize quotes in translations to ASCII straight quotes and strip leading/trailing editorial brackets in translations.

Why this works together:
- After these operations, the three corpora (Competition, Akkademia, ORACC) produce consistent `transliteration_normalized` columns such that the same underlying Akkadian token maps to the same normalized string.
- This reduces vocabulary size, increases effective frequency of tokens shared across corpora, and reduces spurious differences caused purely by editorial or encoding choices.

Preserving linguistic distinctions:
- Diacritics and special glyphs that convey real phonetic distinctions (e.g., `š` vs `s`, `ṣ` vs `s`) are preserved by NFC normalization rather than removed. The pipeline does not strip meaningful diacritics — it standardizes their representation so tokenizers treat them consistently.
- Subscripts indicate sign indices (e.g., different sign variants) and are converted to ASCII numerals but remain as part of the token (e.g., `šar1` vs `šar2`) so the model can still learn different senses if present.

Tokenization guidance (post-normalization):
- Use a subword tokenizer (BPE or SentencePiece) trained on the combined normalized transliteration + translation corpus to learn compact subword vocabularies that respect diacritics and numeric suffixes.
- Keep normalization identical during inference: apply `normalize_transliteration()` and `normalize_translation()` prior to tokenization.

Glossary and proper names:
- Proper names (PN, GN, MN) are extracted into `data/glossary.json`. During decoding you can apply constrained decoding, post-processing lookup, or copy mechanisms to preserve or map names correctly into English output.

Example transformations (representative, not exhaustive):
- `a₁-šur [broken] text!` → normalization → `a1-šur <gap> text`
- `“King”` (fancy quotes) → normalization → `"King"`
- `s` + combining dot (decomposed) → normalization (NFC) → `ṣ`

---

## 4) Practical notes & how to reproduce counts

To see the actual counts and sample normalization examples, run the notebook cell that prints the preprocessing report. A compact snippet to run in the notebook environment:

```python
# Print counts (runs if the notebook variables are in scope)
print('Stage1 rows:', len(stage1_data))
print('Stage2 rows:', len(stage2_data))
print('Stage3 train rows:', len(stage3_train))
print('Stage3 val rows:', len(stage3_val))

# Show normalization example table for competition transliterations
display(get_normalization_samples(train_df, 'transliteration', 'transliteration_normalized', n=10))
```

---

## 5) Recommendations for training & inference

- Always apply the exact same normalization pipeline at inference time.
- Train a joint subword tokenizer (shared source+target) on the normalized corpus.
- Use the 3-stage curriculum as described in the notebook: foundation → domain adaptation → fine-tune.
- Integrate `glossary.json` via post-processing or constrained decoding for improved proper-name handling.

---

If you want, I can:
- Print the exact numeric counts into this file by running the notebook and embedding results here.
- Add example rows (before vs after) directly into this `data.md` (tabular). Just tell me which example sets you want embedded.

