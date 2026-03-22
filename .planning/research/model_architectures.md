# Model Architecture Comparison for Low-Resource Historical Language Translation
# Deep Past: Old Akkadian → English NMT

**Research Date**: 2026-03-22  
**Context**: Old Akkadian cuneiform transliteration → English translation  
**Training Data**: ~200k general + ~80k domain + ~20k competition pairs  
**Strategy**: 3-stage curriculum learning (General → Domain → Competition)

---

## Executive Summary

For the Deep Past Akkadian-English translation task, **mBART-50** emerges as the recommended primary choice, with **mT5-base** as a strong fallback if GPU memory is constrained. NLLB-200-distilled offers compelling advantages for truly low-resource scenarios but lacks explicit ancient language pretraining.

**Key Finding**: mBART-50's multilingual denoising pretraining on 50 languages, combined with proven low-resource MT performance and seq2seq architecture optimized for translation, makes it ideally suited for this 3-stage curriculum learning pipeline. The model's ability to handle specialized notation through proper tokenization and its robustness to domain shift align directly with Akkadian translation challenges.

---

## 1. Architecture Comparison

### 1.1 mBART-50 (facebook/mbart-large-50-many-to-many-mmt)

#### Architecture Specifications
```
Model Type:         Encoder-Decoder Transformer (Seq2Seq)
Total Parameters:   611M (mbart-large-50)
Encoder Layers:     12
Decoder Layers:     12
Attention Heads:    16 (encoder and decoder)
Hidden Size:        1024 (d_model)
FFN Dimension:      4096
Vocabulary Size:    250,027 (multilingual)
Max Sequence:       1024 tokens
Activation:         GELU
```

#### Pretraining Details
- **Corpus**: 50 languages (25 from original mBART + 25 extended)
- **Objective**: Multilingual Denoising Pretraining
  - Text spans masked with Poisson distribution (λ=3.5, 35% masking)
  - Sentence order shuffling for document-level coherence
  - Reconstruction task trains both encoder and decoder jointly
- **Language Representation**: Language ID token prefix (`[lang_code] X [eos]`)
- **Coverage**: Includes low-resource languages (Estonian, Gujarati, Burmese, etc.)

#### Relevance to Low-Resource Akkadian
✅ **Pros**:
1. **Proven Low-Resource Performance**: Designed explicitly for multilingual translation including low-resource language pairs
2. **Full Seq2Seq Pretraining**: Unlike encoder-only models, both generation and comprehension are pretrained
3. **Denoising Robustness**: Masked span prediction trains model to handle incomplete/damaged text (critical for `<gap>` tokens in cuneiform)
4. **Cross-Lingual Transfer**: 50-language pretraining provides strong linguistic priors even for unseen languages
5. **Flexible Fine-Tuning**: Supports multilingual fine-tuning on multiple directions simultaneously

⚠️ **Cons**:
1. **Large Memory Footprint**: 611M parameters require ~16GB GPU for training (batch_size=16 with mixed precision)
2. **No Ancient Languages**: Pretraining corpus lacks historical languages (but this is true for all candidates)
3. **Specialized Tokenization**: Requires careful handling of cuneiform subscripts and diacritics

#### Memory Requirements (FP16 mixed precision)
```
Model Weights:              ~1.2 GB
Optimizer States (AdamW):   ~3.6 GB  (3x model size)
Gradients:                  ~1.2 GB
Activations (batch=16):     ~8-10 GB (depends on sequence length)
--------------------------------------------------
Total Estimated:            ~14-16 GB
```

**Optimization Strategies**:
- Gradient accumulation (4 steps → effective batch_size=64)
- Gradient checkpointing (`use_cache=False` during training)
- DeepSpeed ZeRO Stage 2 for multi-GPU setups

---

### 1.2 mT5-base (google/mt5-base)

#### Architecture Specifications
```
Model Type:         Encoder-Decoder Transformer (T5 framework)
Total Parameters:   580M
Encoder Layers:     12
Decoder Layers:     12
Attention Heads:    12 (encoder and decoder)
Hidden Size:        768
FFN Dimension:      2048
Vocabulary Size:    250,000 (SentencePiece unigram)
Max Sequence:       512 tokens
Activation:         ReLU (relative position bias instead of absolute)
```

#### Pretraining Details
- **Corpus**: mC4 (multilingual Common Crawl) covering 101 languages
- **Objective**: Span-corruption text-to-text pretraining
  - Contiguous spans masked with `<extra_id_*>` sentinel tokens
  - Decoder reconstructs only the masked spans (more efficient than full reconstruction)
- **Text-to-Text Framework**: All tasks cast as text generation (uniform interface)
- **Position Encoding**: Relative position bias (better for variable-length sequences)

#### Relevance to Low-Resource Akkadian
✅ **Pros**:
1. **Broader Language Coverage**: 101 languages (vs mBART's 50) provides richer multilingual representation
2. **Efficient Memory**: Smaller hidden size (768 vs 1024) and FFN (2048 vs 4096) reduce footprint by ~20%
3. **Text-to-Text Flexibility**: Easy to add genre prefix conditioning (`[LETTER] → translation`)
4. **Relative Position Encoding**: Better handles variable-length cuneiform transliterations
5. **Proven NMT Adaptability**: Successfully fine-tuned for many low-resource translation tasks

⚠️ **Cons**:
1. **Less Translation-Specialized**: Pretrained on general language modeling, not specifically MT
2. **Shorter Context**: 512 max tokens (vs mBART's 1024) may truncate longer tablets
3. **No Zero-Shot MT Baseline**: Unlike mBART, requires supervised fine-tuning for any translation

#### Memory Requirements (FP16)
```
Model Weights:              ~1.16 GB
Optimizer States:           ~3.5 GB
Gradients:                  ~1.16 GB
Activations (batch=16):     ~6-8 GB
--------------------------------------------------
Total Estimated:            ~12-14 GB
```

**Optimization Strategies**:
- Lower batch size (8-12) for comfortable training on 12GB GPUs
- Gradient accumulation still recommended for stability

---

### 1.3 NLLB-200-distilled-600M (facebook/nllb-200-distilled-600M)

#### Architecture Specifications
```
Model Type:         Encoder-Decoder (Sparse Mixture of Experts)
Total Parameters:   600M (distilled from 1.3B dense + 54B MoE)
Encoder Layers:     12
Decoder Layers:     12
Architecture:       Based on M2M-100 with conditional compute
Vocabulary Size:    256,000 (SentencePiece)
Max Sequence:       512 tokens (typically)
Languages:          200 languages (including extremely low-resource)
Distillation:       Knowledge distillation from larger NLLB models
```

#### Pretraining Details
- **Corpus**: Custom-mined parallel data for 200 languages (focus on low-resource)
- **Objective**: Multilingual machine translation with Sparsely Gated Mixture of Experts
  - Conditional compute activates relevant experts per language pair
  - Trained on 40,000+ translation directions simultaneously
- **Low-Resource Focus**: Explicit data mining and balancing for underrepresented languages
- **Evaluation**: Flores-200 benchmark (human-translated quality for all 200 languages)

#### Relevance to Low-Resource Akkadian
✅ **Pros**:
1. **Optimized for Low-Resource**: Designed specifically to "leave no language behind"
2. **Distilled Efficiency**: 600M parameters with performance approaching 1.3B dense model
3. **Zero-Shot Transfer**: Strong cross-lingual transfer to truly unseen languages
4. **Robustness**: Toxicity-aware training and extensive low-resource evaluation
5. **Modern Architecture**: MoE conditional compute reduces effective parameter count per forward pass

⚠️ **Cons**:
1. **No Historical Languages**: Like others, lacks ancient language pretraining (Akkadian is ~4000 years removed from training data)
2. **Distillation Trade-offs**: Some capacity lost compared to full dense model
3. **Fixed Tokenizer**: 256k vocabulary may not optimally represent cuneiform transliteration conventions
4. **Longer Inference**: MoE routing adds slight latency (less critical for offline translation)

#### Memory Requirements (FP16)
```
Model Weights:              ~1.2 GB
Optimizer States:           ~3.6 GB
Gradients:                  ~1.2 GB
Activations (batch=16):     ~7-9 GB
--------------------------------------------------
Total Estimated:            ~13-15 GB
```

---

## 2. Architecture Comparison Table

| Feature | mBART-50 | mT5-base | NLLB-200-distilled |
|---------|----------|----------|-------------------|
| **Parameters** | 611M | 580M | 600M |
| **Languages Pretrained** | 50 | 101 | 200 |
| **Low-Resource Focus** | ✓✓✓ High | ✓✓ Medium | ✓✓✓✓ Very High |
| **Translation-Specialized** | ✓✓✓✓ Purpose-built | ✓✓ General purpose | ✓✓✓✓ Purpose-built |
| **Denoising Pretraining** | ✓✓✓✓ Span masking | ✓✓✓ Span corruption | ✓✓ MT-focused |
| **Max Sequence Length** | 1024 | 512 | 512 |
| **Memory Footprint** | ~16GB | ~12GB | ~14GB |
| **Zero-Shot MT Capability** | ✓✓✓ Good | ✗ None | ✓✓✓✓ Excellent |
| **Fine-Tuning Flexibility** | ✓✓✓✓ High | ✓✓✓✓ Very High | ✓✓✓ Good |
| **Curriculum Learning Fit** | ✓✓✓✓ Ideal | ✓✓✓ Good | ✓✓✓ Good |
| **Cuneiform Notation** | ✓✓✓ Custom tok | ✓✓✓ Custom tok | ✓✓ Fixed vocab |
| **HuggingFace Support** | ✓✓✓✓ Excellent | ✓✓✓✓ Excellent | ✓✓✓ Good |
| **Community Precedent** | ✓✓✓✓ Many ancient lang | ✓✓✓ General NMT | ✓✓ Limited ancient |

**Legend**: ✗ = Not applicable, ✓ = Weak, ✓✓ = Fair, ✓✓✓ = Good, ✓✓✓✓ = Excellent

---

## 3. Curriculum Learning Best Practices

### 3.1 Progressive Fine-Tuning Strategy

#### Learning Rate Schedules
```python
# Recommended for 3-stage curriculum
STAGE_1_CONFIG = {
    'lr': 1e-4,           # Higher for foundational learning
    'warmup_steps': 1000,
    'scheduler': 'linear_decay',
    'min_lr': 1e-6,
    'epochs': 10
}

STAGE_2_CONFIG = {
    'lr': 5e-5,           # 50% reduction to preserve Stage 1 knowledge
    'warmup_steps': 500,
    'scheduler': 'linear_decay',
    'min_lr': 5e-7,
    'epochs': 5
}

STAGE_3_CONFIG = {
    'lr': 1e-5,           # 80% reduction for precise fine-tuning
    'warmup_steps': 200,
    'scheduler': 'cosine',  # Cosine for smooth convergence
    'min_lr': 1e-7,
    'epochs': 10,
    'early_stopping_patience': 3
}
```

**Rationale**: Progressive LR decay prevents catastrophic forgetting. Each stage builds on previous knowledge with increasingly conservative updates.

#### Layer Freezing Strategies

**Option A: No Freezing (Recommended for Akkadian)**
- **Why**: Ancient language is far from pretraining distribution; all layers need adaptation
- **Risk**: Higher chance of catastrophic forgetting
- **Mitigation**: Very low LR in later stages + gradient clipping

**Option B: Gradual Unfreezing (Alternative)**
```python
# Stage 1: Unfreeze all (foundational learning)
# Stage 2: Freeze bottom 6 encoder layers
for param in model.model.encoder.layers[:6].parameters():
    param.requires_grad = False

# Stage 3: Unfreeze all again for competition-specific tuning
for param in model.parameters():
    param.requires_grad = True
```

### 3.2 Preventing Catastrophic Forgetting

#### Technique 1: Data Mixing
```python
# Mix previous stage data into current stage
STAGE_2_DATA_MIX = {
    'stage2_data': 0.8,   # 80% OARE domain data
    'stage1_sample': 0.2  # 20% random sample from Stage 1 (Akkademia+ORACC)
}

STAGE_3_DATA_MIX = {
    'stage3_data': 0.7,   # 70% competition data
    'stage2_sample': 0.2, # 20% OARE samples
    'stage1_sample': 0.1  # 10% general Akkadian
}
```

**Implementation**:
```python
def create_mixed_dataset(primary_path, auxiliary_paths, mix_ratios):
    primary_df = pd.read_csv(primary_path)
    primary_size = int(len(primary_df) * mix_ratios[0])
    
    datasets = [primary_df.sample(primary_size, random_state=42)]
    
    for aux_path, ratio in zip(auxiliary_paths, mix_ratios[1:]):
        aux_df = pd.read_csv(aux_path)
        sample_size = int(len(primary_df) * ratio)
        datasets.append(aux_df.sample(sample_size, replace=True, random_state=42))
    
    return pd.concat(datasets, ignore_index=True).sample(frac=1, random_state=42)
```

#### Technique 2: Elastic Weight Consolidation (EWC)
```python
# After Stage 1, compute Fisher Information Matrix
def compute_fisher(model, dataloader, device):
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    model.eval()
    for batch in dataloader:
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            fisher[name] += param.grad.data ** 2 / len(dataloader)
    
    return fisher

# During Stage 2/3 training, add EWC regularization
def ewc_loss(model, fisher, prev_params, lambda_ewc=0.1):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - prev_params[name]) ** 2).sum()
    return lambda_ewc * loss
```

#### Technique 3: Knowledge Distillation
```python
# Use Stage 1 model as "teacher" for Stage 2
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    """
    Combines cross-entropy loss with KL divergence from teacher
    alpha: weight for distillation loss (1-alpha for CE loss)
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return alpha * kd_loss + (1 - alpha) * ce_loss
```

### 3.3 Optimal Stopping Criteria Per Stage

#### Stage 1: Foundation
**Goal**: Broad linguistic coverage, not competition-specific
```python
STAGE1_STOPPING = {
    'primary_metric': 'validation_loss',
    'patience': 3,          # Stop if no improvement for 3 epochs
    'min_delta': 0.01,      # Minimum change to count as improvement
    'validation_freq': 1,   # Validate every epoch
    'save_strategy': 'best_loss'  # Not BLEU (not competition domain)
}
```

#### Stage 2: Domain Adaptation
**Goal**: OARE-style translation, monitor both loss and BLEU
```python
STAGE2_STOPPING = {
    'primary_metric': 'validation_bleu',
    'secondary_metric': 'validation_loss',
    'patience': 2,          # Shorter patience (smaller dataset)
    'min_delta': 0.5,       # BLEU improvement threshold
    'validation_freq': 1,
    'save_strategy': 'best_bleu'
}
```

#### Stage 3: Competition Fine-Tuning
**Goal**: Maximize validation BLEU on held-out competition split
```python
STAGE3_STOPPING = {
    'primary_metric': 'validation_bleu',
    'secondary_metrics': ['chrf', 'ter'],  # Track multiple metrics
    'patience': 3,          # Allow overfitting slightly (small dataset)
    'min_delta': 0.3,       # Lower threshold for small improvements
    'validation_freq': 1,   # Check every epoch
    'save_strategy': 'best_bleu',
    'save_total_limit': 5,  # Keep top 5 checkpoints
    'load_best_at_end': True  # Revert to best checkpoint
}
```

---

## 4. Low-Resource NMT Techniques for Historical Languages

### 4.1 Data Augmentation Strategies

#### Technique 1: Back-Translation
**Feasibility for Akkadian**: ⚠️ Limited (no reverse model exists initially)

**Workaround**:
1. Train initial Akkadian→English model (Stage 1+2)
2. Train reverse English→Akkadian model on same data
3. Generate synthetic Akkadian from monolingual English corpus
4. Filter by confidence score (keep only high-quality back-translations)

```python
# Pseudo-code for back-translation
def generate_synthetic_pairs(en_sentences, en2ak_model, threshold=0.8):
    synthetic_pairs = []
    for en in tqdm(en_sentences):
        ak_pred = en2ak_model.generate(en, num_beams=5, return_scores=True)
        if ak_pred.score > threshold:  # Confidence filtering
            synthetic_pairs.append((ak_pred.text, en))
    return synthetic_pairs

# Add to Stage 2 as additional data (mix 10-20% synthetic)
```

**Recommendation**: **Skip back-translation for Akkadian** due to:
- No monolingual Akkadian corpus available
- Risk of amplifying errors from reverse model
- Limited benefit vs complexity for ~200k supervised pairs

#### Technique 2: Source-Side Noise Injection
**Rationale**: Ancient texts have transcription variations, damage, and editorial uncertainty

```python
def add_cuneiform_noise(text, p_drop=0.05, p_gap=0.02, p_sub=0.03):
    """Simulate real-world cuneiform transcription variability"""
    tokens = text.split()
    noisy_tokens = []
    
    for token in tokens:
        r = random.random()
        if r < p_drop:
            continue  # Simulate missing text
        elif r < p_drop + p_gap:
            noisy_tokens.append('<gap>')  # Simulate damage
        elif r < p_drop + p_gap + p_sub:
            # Simulate subscript variation (a1 ↔ a2)
            noisy_tokens.append(vary_subscript(token))
        else:
            noisy_tokens.append(token)
    
    return ' '.join(noisy_tokens)

# Apply during Stage 1 training only (builds robustness)
```

#### Technique 3: Genre-Specific Paraphrasing
```python
GENRE_PARAPHRASES = {
    '[LETTER]': ['[EPISTLE]', '[CORRESPONDENCE]'],  # Vary genre tags
    '[DEBT_NOTE]': ['[LOAN]', '[CREDIT]'],
    '[CONTRACT]': ['[AGREEMENT]', '[COMPACT]']
}

def augment_with_genre_variations(df):
    """Duplicate competition data with alternative genre tags"""
    augmented = []
    for _, row in df.iterrows():
        augmented.append(row)
        for genre, paraphrases in GENRE_PARAPHRASES.items():
            if row['transliteration'].startswith(genre):
                for para in paraphrases:
                    new_row = row.copy()
                    new_row['transliteration'] = new_row['transliteration'].replace(genre, para, 1)
                    augmented.append(new_row)
    return pd.DataFrame(augmented)
```

### 4.2 Handling Specialized Notation

#### Subscripts and Diacritics
**Challenge**: Cuneiform transliteration uses Unicode subscripts (₁₂₃), special diacritics (šṣṭ), and indices (a1, a2)

**Solution 1: Normalize to ASCII (Recommended)**
```python
# Already implemented in preprocessing pipeline
def normalize_transliteration(text):
    # Subscripts to ASCII digits
    text = text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    # Preserve diacritics (š, ṣ, etc.) via NFC normalization
    text = unicodedata.normalize('NFC', text)
    return text
```

**Why**: Reduces vocabulary size, improves tokenization efficiency, no semantic loss

**Solution 2: Custom SentencePiece Tokenizer**
```python
# Train domain-specific tokenizer on normalized Akkadian
import sentencepiece as spm

# Combine all Akkadian transliterations
with open('akkadian_corpus.txt', 'w') as f:
    for df in [stage1_df, stage2_df, stage3_df]:
        f.write('\n'.join(df['transliteration_normalized']) + '\n')

# Train unigram tokenizer
spm.SentencePieceTrainer.train(
    input='akkadian_corpus.txt',
    model_prefix='akkadian_sp',
    vocab_size=8000,  # Smaller vocab for specialized domain
    character_coverage=0.9995,  # High coverage for all diacritics
    model_type='unigram',
    user_defined_symbols=['<gap>', '[LETTER]', '[DEBT_NOTE]', '[CONTRACT]', '[ADMIN]'],
    unk_piece='<unk>',
    pad_piece='<pad>',
    normalization_rule_name='nfkc'  # Unicode normalization
)
```

#### Lacunae Markers (`<gap>` tokens)
**Training Strategy**:
1. **Stage 1**: Full `<gap>` exposure (ORACC contains many damaged texts)
2. **Stage 2**: Moderate `<gap>` (OARE has some damage markers)
3. **Stage 3**: Realistic `<gap>` distribution (competition data)

**Model Adaptation**:
```python
# Add <gap> as special token to tokenizer
tokenizer.add_special_tokens({'additional_special_tokens': ['<gap>']})
model.resize_token_embeddings(len(tokenizer))

# During training, teach model to output "..." or omit for <gap>
# Example training pair:
# Source: "a-na <gap> šar-ri"
# Target: "to ... the king"  OR  "to the king"  (both acceptable)
```

### 4.3 Proper Name Handling

#### Strategy 1: Glossary Post-Processing (Recommended)
```python
def apply_glossary(source, translation, glossary_dict):
    """Replace model translations with canonical proper names"""
    for ak_form, canonical_info in glossary_dict.items():
        if ak_form in source:  # If source contains proper name
            # Replace any model-generated variants with canonical form
            for variant in canonical_info['variants']:
                translation = re.sub(
                    rf'\b{re.escape(variant)}\b', 
                    canonical_info['canonical'], 
                    translation, 
                    flags=re.IGNORECASE
                )
    return translation

# Example:
# Source: "ha-am-mu-ra-pi" → Glossary: "Hammurabi"
# Model output: "Hammurapi" → Post-processed: "Hammurabi"
```

#### Strategy 2: Constrained Decoding
```python
from transformers import LogitsProcessor

class GlossaryBiasProcessor(LogitsProcessor):
    def __init__(self, glossary_token_ids, bias_weight=5.0):
        self.glossary_ids = glossary_token_ids
        self.bias_weight = bias_weight
    
    def __call__(self, input_ids, scores):
        # Boost probability of glossary tokens during generation
        for token_id in self.glossary_ids:
            scores[:, token_id] += self.bias_weight
        return scores

# Use during inference
outputs = model.generate(
    input_ids,
    logits_processor=[GlossaryBiasProcessor(glossary_ids)],
    num_beams=5
)
```

#### Strategy 3: Copy Mechanism (Advanced)
**Not recommended for this project** (requires custom model architecture changes)

### 4.4 Evaluation Metrics for Historical Texts

#### Standard MT Metrics
```python
import sacrebleu
from evaluate import load

# BLEU (primary metric for competition)
bleu = sacrebleu.corpus_bleu(predictions, [references])

# chrF++ (better for morphologically rich languages)
chrf = sacrebleu.corpus_chrf(predictions, [references])

# TER (Translation Edit Rate - lower is better)
ter_metric = load('ter')
ter = ter_metric.compute(predictions=predictions, references=references)
```

#### Domain-Specific Metrics
```python
def proper_name_accuracy(pred, ref, glossary):
    """Measure accuracy on proper names specifically"""
    pred_names = extract_proper_names(pred, glossary)
    ref_names = extract_proper_names(ref, glossary)
    
    correct = len(pred_names & ref_names)
    total = len(ref_names)
    return correct / total if total > 0 else 0.0

def gap_handling_score(pred, src):
    """Evaluate how well model handles <gap> tokens"""
    gap_count = src.count('<gap>')
    if gap_count == 0:
        return 1.0
    
    # Check if prediction omits or indicates missing text
    ellipsis_count = pred.count('...') + pred.count('[...]')
    return min(ellipsis_count / gap_count, 1.0)
```

#### BLEU Limitations for Akkadian
⚠️ **Known Issues**:
1. **Multiple Valid Translations**: Ancient texts have interpretive flexibility
2. **Proper Name Variants**: "Hammurabi" vs "Hammurapi" penalized as errors
3. **Genre Conventions**: Letters have formulaic phrases with acceptable variations

**Mitigation**:
- Use chrF++ as secondary metric (character-level n-grams)
- Manual evaluation of 100-sample subset per stage
- Report per-genre BLEU scores separately

---

## 5. Concrete Recommendations

### 5.1 Primary Recommendation: mBART-50

**Why mBART-50 for Akkadian Translation?**

1. **Translation-Specialized Architecture**: Pretrained specifically for multilingual MT, not general language modeling
2. **Denoising Robustness**: Masked span prediction directly addresses `<gap>` token handling
3. **Low-Resource Proven**: Extensive use in historical and ancient language projects (Latin, Ancient Greek, Sanskrit)
4. **Curriculum Learning Fit**: Supports progressive fine-tuning with minimal catastrophic forgetting
5. **Longer Context**: 1024 tokens accommodates full tablet transcriptions
6. **Community Support**: Abundant fine-tuning examples and troubleshooting resources

**Implementation Timeline**:
```
Week 1: Custom tokenizer training (8k vocab with cuneiform notation)
Week 2: Stage 1 training (10 epochs, ~24 GPU hours)
Week 3: Stage 2 training (5 epochs, ~12 GPU hours)
Week 4: Stage 3 training + glossary integration (10 epochs + post-processing)
```

**Hardware Requirements**:
- **Minimum**: 1x RTX 3090 (24GB) or A5000
- **Optimal**: 2x A6000 (48GB) with DeepSpeed for faster training
- **Budget**: Google Colab Pro+ or Lambda Labs GPU rental (~$1-2/hour)

### 5.2 Fallback Recommendation: mT5-base

**When to Choose mT5-base?**
- GPU memory limited to 12-16GB (RTX 3080/4070 Ti)
- Preference for text-to-text framework flexibility
- Budget constraints (20% faster training)

**Trade-offs**:
- Slightly lower expected BLEU (~2-3 points based on literature)
- Requires more careful hyperparameter tuning
- Max 512 tokens may truncate longest tablets

### 5.3 Experimental Alternative: NLLB-200-distilled

**When to Consider NLLB?**
- Research-focused comparison (not production)
- Investigating zero-shot transfer from related ancient languages (Sumerian, Aramaic)
- Expecting test set to have radically different distribution

**Not Recommended Because**:
- Less flexible tokenization (fixed 256k vocab)
- Fewer fine-tuning examples for curriculum learning
- No proven advantage over mBART for similar low-resource scenarios

---

## 6. Training Configuration Recommendations

### 6.1 Stage 1: General Akkadian Foundation
```python
STAGE1_CONFIG = {
    # Model
    'model_name': 'facebook/mbart-large-50-many-to-many-mmt',
    'tokenizer': 'custom_akkadian_sp_8k',  # Train custom tokenizer
    
    # Data
    'train_data': 'data/stage1_train.csv',  # ~200k pairs
    'data_mix': {'akkademia': 0.3, 'oracc': 0.7},
    'max_source_length': 512,
    'max_target_length': 512,
    
    # Training
    'epochs': 10,
    'batch_size': 16,
    'gradient_accumulation_steps': 4,  # Effective batch=64
    'learning_rate': 1e-4,
    'warmup_steps': 1000,
    'lr_scheduler': 'linear',
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # Regularization
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'label_smoothing': 0.1,
    
    # Optimization
    'fp16': True,  # Mixed precision
    'gradient_checkpointing': True,  # Reduce memory
    'dataloader_num_workers': 4,
    
    # Evaluation
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
    'save_total_limit': 3,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'loss',  # Not BLEU (different domain)
    
    # Logging
    'logging_steps': 100,
    'report_to': 'tensorboard'
}
```

### 6.2 Stage 2: OARE Domain Adaptation
```python
STAGE2_CONFIG = {
    # Model
    'checkpoint': 'models/stage1_final',  # Load from Stage 1
    
    # Data
    'train_data': 'data/stage2_train.csv',  # ~80k OARE pairs
    'data_mix': {'stage2': 0.8, 'stage1_sample': 0.2},  # Mix to prevent forgetting
    
    # Training (lower LR to preserve Stage 1 knowledge)
    'epochs': 5,
    'batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 5e-5,  # 50% of Stage 1
    'warmup_steps': 500,
    'lr_scheduler': 'linear',
    
    # Regularization (slightly higher to prevent OARE overfitting)
    'label_smoothing': 0.15,
    'weight_decay': 0.015,
    
    # Evaluation
    'metric_for_best_model': 'bleu',  # OARE domain, can use BLEU
    'early_stopping_patience': 2
}
```

### 6.3 Stage 3: Competition Fine-Tuning
```python
STAGE3_CONFIG = {
    # Model
    'checkpoint': 'models/stage2_final',
    
    # Data
    'train_data': 'data/stage3_train.csv',  # ~20k competition pairs
    'val_data': 'data/stage3_val.csv',      # ~2k validation
    'data_mix': {'stage3': 0.7, 'stage2_sample': 0.2, 'stage1_sample': 0.1},
    'apply_genre_prefixes': True,  # [LETTER], [DEBT_NOTE], etc.
    
    # Training (very low LR for precise tuning)
    'epochs': 10,
    'batch_size': 8,  # Smaller batch for small dataset
    'gradient_accumulation_steps': 8,  # Effective batch=64
    'learning_rate': 1e-5,  # 80% reduction from Stage 2
    'warmup_steps': 200,
    'lr_scheduler': 'cosine',  # Smooth convergence
    
    # Regularization (careful with small dataset)
    'dropout': 0.15,  # Higher dropout to prevent overfitting
    'label_smoothing': 0.1,
    'weight_decay': 0.02,
    
    # Evaluation (critical for small validation set)
    'eval_strategy': 'epoch',
    'metric_for_best_model': 'bleu',
    'greater_is_better': True,
    'early_stopping_patience': 3,
    'save_total_limit': 5,  # Keep top 5 checkpoints
    
    # Generation parameters (for BLEU computation)
    'predict_with_generate': True,
    'generation_max_length': 512,
    'generation_num_beams': 5
}
```

### 6.4 Inference Configuration
```python
INFERENCE_CONFIG = {
    'checkpoint': 'models/stage3_final',
    'batch_size': 32,  # Larger batch for inference
    'num_beams': 5,
    'early_stopping': True,
    'max_length': 512,
    'length_penalty': 1.0,  # No length bias
    'no_repeat_ngram_size': 3,  # Prevent repetition
    'apply_glossary': True,  # Post-process proper names
    'glossary_path': 'data/glossary.json'
}
```

---

## 7. Success Metrics and Benchmarks

### 7.1 Quantitative Targets

| Stage | Validation BLEU | chrF++ | TER | Notes |
|-------|-----------------|--------|-----|-------|
| **Stage 1** | ≥20 | ≥45 | ≤70 | Low-resource baseline (general Akkadian) |
| **Stage 2** | +3 to +5 | +5 to +8 | -5 to -8 | Domain adaptation improvement |
| **Stage 3** | ≥30 | ≥55 | ≤60 | Competitive performance on competition val |

**Rationale**:
- BLEU 20 is typical for low-resource MT from scratch
- Stage 2 should show clear improvement on OARE-style texts
- Stage 3 target of 30 BLEU is realistic for ~20k fine-tuning pairs

### 7.2 Qualitative Checks

**Per-Stage Manual Review** (100 samples each):
1. **Proper Name Accuracy**: ≥85% with glossary post-processing
2. **Gap Handling**: Model outputs "..." or omits text for `<gap>` tokens
3. **Genre Conditioning**: [LETTER] produces epistolary style, [CONTRACT] produces legal formulations
4. **Grammatical Coherence**: Translations are fluent English (not word-by-word)
5. **No Hallucination**: Model doesn't invent content not in source

### 7.3 Diagnostic Metrics

```python
# Implement per-genre BLEU
def evaluate_by_genre(df, model, tokenizer):
    results = {}
    for genre in ['[LETTER]', '[DEBT_NOTE]', '[CONTRACT]', '[ADMIN]']:
        genre_df = df[df['transliteration'].str.startswith(genre)]
        if len(genre_df) > 0:
            predictions = generate_translations(model, tokenizer, genre_df)
            bleu = compute_bleu(predictions, genre_df['translation'])
            results[genre] = bleu
    return results

# Track catastrophic forgetting
def measure_forgetting(model, stage1_val_df):
    """Compare Stage 2/3 model performance on Stage 1 validation"""
    bleu_stage2 = evaluate_model(model_stage2, stage1_val_df)
    bleu_stage1 = evaluate_model(model_stage1, stage1_val_df)
    forgetting = bleu_stage1 - bleu_stage2
    if forgetting > 5:
        print(f"WARNING: Catastrophic forgetting detected ({forgetting:.1f} BLEU drop)")
    return forgetting
```

---

## 8. Risk Mitigation

### 8.1 GPU Memory Issues

**Symptoms**: OOM errors during training

**Solutions**:
1. **Immediate**: Reduce batch_size to 8, increase gradient_accumulation to 8
2. **Short-term**: Enable gradient checkpointing (`use_cache=False`)
3. **Medium-term**: Switch to mT5-base (~20% memory reduction)
4. **Long-term**: Rent larger GPU (A6000 48GB) or multi-GPU setup

### 8.2 Catastrophic Forgetting

**Symptoms**: Stage 1 validation BLEU drops >5 points after Stage 2/3

**Solutions**:
1. **Data Mixing**: Add 20% Stage 1 samples to Stage 2 training
2. **Lower LR**: Reduce Stage 2/3 learning rates further (3e-5, 5e-6)
3. **EWC Regularization**: Implement Fisher-weighted penalty (see Section 3.2)
4. **Checkpoint Rollback**: Use Stage 1 model with only Stage 3 fine-tuning

### 8.3 Low BLEU Scores

**Symptoms**: Competition validation BLEU <20

**Root Causes**:
1. **Proper Names**: BLEU penalizes name variants heavily
2. **Formulaic Phrases**: Multiple valid translations for fixed expressions
3. **Genre Mismatch**: Training genres don't match test distribution

**Solutions**:
1. **Glossary Post-Processing**: Apply proper name corrections (expected +3-5 BLEU)
2. **Ensemble**: Average predictions from top 3 checkpoints (expected +1-2 BLEU)
3. **Human Evaluation**: Request manual review of 100 samples (BLEU may underestimate quality)

### 8.4 Overfitting on Small Stage 3 Dataset

**Symptoms**: Training BLEU >>validation BLEU, early stopping triggers at epoch 3-4

**Solutions**:
1. **Higher Dropout**: Increase to 0.2 (from 0.15)
2. **Data Augmentation**: Genre paraphrasing, noise injection (see Section 4.1)
3. **Reduce Epochs**: Stop at 7 epochs instead of 10
4. **Regularization**: Increase weight_decay to 0.03

---

## 9. Implementation Checklist

### Phase 1: Model Setup (Week 1)
- [ ] Install dependencies (`transformers==4.40.0`, `sentencepiece`, `sacrebleu`)
- [ ] Download mBART-50 checkpoint (`facebook/mbart-large-50-many-to-many-mmt`)
- [ ] Test model loading and basic inference on sample text
- [ ] Train custom SentencePiece tokenizer on normalized Akkadian corpus (8k vocab)
- [ ] Verify tokenizer handles `<gap>`, genre tags, subscripts correctly
- [ ] Resize model embeddings to accommodate new special tokens
- [ ] Benchmark memory usage (batch_size=16, seq_len=512)

### Phase 2: Stage 1 Training (Week 2)
- [ ] Load and verify `data/stage1_train.csv` (~200k pairs)
- [ ] Implement AkkadianDataset class with proper tokenization
- [ ] Set up training loop with mixed precision, gradient checkpointing
- [ ] Configure TensorBoard logging
- [ ] Train for 10 epochs with LR=1e-4
- [ ] Monitor training/validation loss curves
- [ ] Evaluate on Stage 1 validation set (expect BLEU ≥20)
- [ ] Save best checkpoint to `models/stage1_final/`
- [ ] Generate 100 sample translations for qualitative review

### Phase 3: Stage 2 Training (Week 3)
- [ ] Load Stage 1 checkpoint
- [ ] Prepare Stage 2 data with 20% Stage 1 mixing
- [ ] Train for 5 epochs with LR=5e-5
- [ ] Evaluate on OARE validation set (expect +3-5 BLEU improvement)
- [ ] Check catastrophic forgetting on Stage 1 val (should be <2 BLEU drop)
- [ ] Save checkpoint to `models/stage2_final/`

### Phase 4: Stage 3 Training (Week 4)
- [ ] Load Stage 2 checkpoint
- [ ] Verify genre prefixes applied to competition data
- [ ] Prepare data mixing (70% Stage 3, 20% Stage 2, 10% Stage 1)
- [ ] Train for up to 10 epochs with LR=1e-5, early stopping (patience=3)
- [ ] Evaluate on `data/stage3_val.csv` every epoch
- [ ] Select best checkpoint based on validation BLEU
- [ ] Implement glossary post-processing
- [ ] Measure proper name accuracy on validation set
- [ ] Save final model to `models/stage3_final/`

### Phase 5: Inference & Submission
- [ ] Generate predictions for test set (`dataset/test.csv`)
- [ ] Apply glossary corrections
- [ ] Format as submission CSV
- [ ] Sanity check: no missing translations, valid Unicode
- [ ] Generate confidence scores for low-quality predictions
- [ ] Save to `outputs/submission.csv`

---

## 10. References and Further Reading

### Primary Research Papers
1. **mBART-50**: Tang et al. (2020). "Multilingual Translation with Extensible Multilingual Pretraining and Finetuning." arXiv:2008.00401
2. **mT5**: Xue et al. (2020). "mT5: A massively multilingual pre-trained text-to-text transformer." arXiv:2010.11934
3. **NLLB-200**: NLLB Team (2022). "No Language Left Behind: Scaling Human-Centered Machine Translation." arXiv:2207.04672

### Curriculum Learning
4. Bengio et al. (2009). "Curriculum Learning." ICML 2009
5. Ruder & Plank (2017). "Learning to select data for transfer learning with Bayesian Optimization." EMNLP 2017
6. Kocmi & Bojar (2017). "Curriculum Learning and Minibatch Bucketing in Neural Machine Translation." RANLP 2017

### Low-Resource NMT
7. Sennrich et al. (2016). "Improving Neural Machine Translation Models with Monolingual Data." ACL 2016 (Back-translation)
8. Gu et al. (2018). "Meta-Learning for Low-Resource Neural Machine Translation." EMNLP 2018
9. Arivazhagan et al. (2019). "Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges." arXiv:1907.05019

### Historical Language NMT
10. Bollmann (2019). "A Large-Scale Comparison of Historical Text Normalization Systems." NAACL 2019
11. Varga & Simon (2007). "Parallel corpora for medium density languages." RANLP 2007

### HuggingFace Documentation
12. mBART Model Card: https://huggingface.co/facebook/mbart-large-50
13. mT5 Model Card: https://huggingface.co/google/mt5-base
14. Transformers Training Guide: https://huggingface.co/docs/transformers/training

---

## Appendix A: Compute Cost Estimation

### Training Time Estimates (single RTX 3090)

| Stage | Epochs | Examples | Time per Epoch | Total Time |
|-------|--------|----------|----------------|------------|
| Stage 1 | 10 | 200,000 | 2.5 hours | ~25 hours |
| Stage 2 | 5 | 80,000 | 1 hour | ~5 hours |
| Stage 3 | 10 | 20,000 | 20 minutes | ~3.5 hours |
| **Total** | - | - | - | **~34 hours** |

**Cloud GPU Costs** (Lambda Labs pricing):
- RTX 3090 ($0.50/hour): $17 total
- A6000 ($0.80/hour): $27 total
- 2x A6000 ($1.60/hour, 2x speedup): $27 total (faster)

### Inference Time
- Test set: ~5,000 examples
- Inference: ~10 minutes (batch_size=32, num_beams=5)
- Negligible cost

---

## Appendix B: Alternative Approaches Not Recommended

### 1. Encoder-Only Models (BERT, XLM-R)
**Why Not**: No generation capability; would require separate decoder

### 2. Decoder-Only Models (GPT, LLaMA)
**Why Not**: Inefficient for translation; require large prompts; inferior to encoder-decoder for MT

### 3. Training from Scratch
**Why Not**: ~200k pairs insufficient for high-quality model; pretraining provides crucial linguistic priors

### 4. Character-Level Models
**Why Not**: Cuneiform transliteration is already phonetic representation; subword tokenization more efficient

### 5. External Retrieval-Augmented Generation
**Why Not**: No large Akkadian reference corpus available; adds complexity without clear benefit

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-22  
**Maintainer**: Deep Past Challenge Team  
**Review Cycle**: Update after Stage 1/2/3 training completion with empirical results
