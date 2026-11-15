**English** | [Русский](README.ru.md)

# Named Entity Recognition for Russian News (NEREL): CRF and a Compact Transformer under Hard Constraints

A course project (ISP RAS practicum, autumn 2026). The task is to find named entities of 29 types (PERSON, ORGANIZATION,
PROFESSION, EVENT, DATE, etc.) in Russian news texts, including **nested** entities, and return their boundaries and
types. The solution is submitted to an automated grader with strict limits, so the work is not only about training a
model but also about engineering within a size and time budget.

**Outcome:** the final solution is a fine-tuned `rubert-tiny2` with a shared encoder and three heads for nested
entities, with the embedding matrix compressed to fit the 16 MB limit. It reaches **macro-F1 0.5994** on the hidden test
set (precision 0.603, recall 0.609), +0.035 over the best CRF (0.5645) and +0.065 over my first transformer (0.5346).
The path there (flat CRF → nested layers → a transformer that first lost to the CRF → hypothesis tests that explained
why) is documented below.

## Task and constraints

| | |
|---|---|
| Data | NEREL v1.1 corpus (Russian news, 29 entity types, many nested entities) |
| Metric | micro-F1 per entity type, averaged over types (macro over types); exact match of boundaries and type |
| Solution size | archive ≤ 16 MB |
| Runtime | ≤ 30 minutes on the test set, 8 CPUs, 16 GB RAM, **no internet access** |
| Deliverables | `solution.py` with a `Solution` class, a method description, training code and reproduction instructions |
| Rules | the test split must not be used for training or tuning; at most 7 submissions per week |

Because the metric averages over types, rare types (LAW, CRIME, RELIGION, FAMILY, DISTRICT, ...) matter as much as
frequent ones (PERSON, DATE). This shaped many of the decisions below.

## Results on the hidden test set (grader)

| # | Solution | Configuration | Macro-F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1 | CRF, single layer (flat BIO tagging) | first variant: c1=0.1, c2=0.05, min-freq=3, 150 iterations; parameters changed in the other iterations (details were not recorded) | 0.445-0.465 (best 0.4646) | 0.600 | 0.397 |
| 2 | CRF, **3 nesting layers**, trained on train | c1=0.05, c2=0.02, min-freq=2, 150 iterations; features with stems and bigrams | 0.5469 | 0.631 | 0.492 |
| 3 | same, trained on train+dev | same parameters | 0.5645 | 0.659 | 0.509 |
| 4 | rubert-tiny2, 3 heads (one per layer), pruned vocabulary | train+dev, 9 epochs; everything else default (encoder lr 1e-4, head lr 1e-3, batch 32, vocabulary: 10k most frequent + short pieces) | 0.4956 | 0.557 | 0.484 |
| 5 | rubert-tiny2, pruned vocabulary | train+dev, 20 epochs; everything else default (as in row 4) | 0.5117 | 0.550 | 0.508 |
| 6 | rubert-tiny2, pruned vocabulary | train+dev, 22 epochs, encoder lr 3e-4, batch 16, vocabulary: 12k most frequent + short pieces | 0.5346 | 0.585 | 0.518 |
| 7 | **rubert-tiny2, final** | train+dev, 13 epochs, encoder lr 3e-4, batch 16; all Cyrillic/Latin vocabulary pieces (83,303), rank-96 factorised int8 token embeddings, word-casing embedding | **0.5994** | 0.603 | 0.609 |

Local validation (dev, not used for training) agrees reasonably with the test set: the layered CRF scored 0.560 on dev
and 0.547 on test. Prediction time on the grader is about 1-2 seconds for every solution.

## What I did and why

### 1. Baseline and the ceiling of the formulation
The first version is a linear-chain CRF (`sklearn-crfsuite`) with hand-crafted features (word, prefixes/suffixes, word
shape, casing, neighbours in a ±2 window) on flat BIO tags. Result: 0.469 on dev.

Before improving the model, I computed the **ceiling of the formulation itself**: even a perfect labelling of dev, if
encoded as flat BIO, gives a macro-F1 of only **0.854**. The reason is nested entities (for example, "Russia" inside
"Government of Russia"): flat tagging loses them by construction. About 0.15 of the score was therefore lost before the
model even mattered.

### 2. Nesting layers
Entities are sorted by length and greedily assigned to three layers so that spans within a layer do not overlap
(outer → nested → nested-in-nested; the train set has 29,709 / 6,507 / 1,019 entities per layer). A separate CRF is
trained per layer (in parallel) and the predictions are merged. The labelling ceiling rose from 0.854 to **0.978**.
Effect on the test set: **+0.08** macro-F1 (0.465 → 0.547). Training on train+dev added another +0.018. The same
layering is used later by the transformer, with one classification head per layer.

### 3. Error analysis and decoding experiments
In the CRF tables precision is noticeably higher than recall, i.e. the model is conservative. I tried biased decoding
(choosing the tag from marginal probabilities with a threshold on the `O` class): on dev it did not help
(0.560 → 0.556 and lower), so it was left out. This negative result is documented to avoid revisiting the idea.

### 4. First transformer attempt under the 16 MB limit
To improve rare types, I fine-tuned the pretrained `cointegrated/rubert-tiny2` (3 layers, hidden size 312). The main
engineering problem is size: the model's vocabulary (83,828 tokens) accounts for most of its weight. The first version
pruned the vocabulary to special tokens, short pieces (≤ 2 characters) and the most frequent tokens of the training
texts (10-12k); all other tokens mapped to `[UNK]`. Weights were stored in fp16 (6.4M parameters, about 12 MB).

It was worse than the CRF: 0.4956 → 0.5117 → 0.5346 on the grader after more epochs and tuning, still below 0.5645. On
dev (without training on it) it plateaued around 0.52 after about 10 epochs while the training loss kept falling (in
later runs F1 on a training subset reached ≈0.98 against ≈0.53 on dev), so the problem was generalisation, not
underfitting. By type it lost to the CRF almost everywhere, especially on PERSON (F1 0.71 vs 0.85), that is, on names.

### 5. Testing two hypotheses
I formulated two hypotheses and tested them with `diagnose_vocab.py` and controlled runs on dev (trained on train only,
seed 42, 15 epochs, encoder lr 3e-4, batch 16):

| Run | Vocabulary | Casing embedding | Token embeddings | Dev macro-F1: best epoch (epoch 15) | PERSON F1 (best epoch) |
|---|---|---|---|---|---|
| A | pruned, 12k frequent tokens | no | full | 0.5300 (0.5291) | 0.721 |
| B | pruned, 12k frequent tokens | yes | full | 0.5565 (0.5500) | 0.805 |
| C | full (83,828 tokens) | no | full | 0.6105 (0.6068) | 0.868 |
| E | full (83,828 tokens) | yes | full | 0.6134 (0.6134) | 0.875 |
| D | full (83,828 tokens) | yes | rank-64 factorisation | 0.5862 (0.5862) | 0.857 |
| F | Cyrillic/Latin only (83,303) | yes | rank-96 factorisation | 0.6280 (0.6073) | 0.864 |
| G | Cyrillic/Latin only (83,303) | yes | rank-128 factorisation | 0.6107 (0.6085) | 0.858 |
| - | CRF, 3 layers (reference) | | | 0.5600 | 0.850 |

Runs C and E keep the full ≈55 MB (fp16) embedding matrix and are therefore only diagnostic; the factorised runs
(D, F, G) are the ones that can be exported into the 16 MB limit.

**Hypothesis 2 (vocabulary pruning) - confirmed.** With the 12k pruned vocabulary, 19.7% of word pieces on dev map to
`[UNK]`, and 43% of all gold entities (66% of PERSON entities) contain at least one `[UNK]`. Keeping the whole
vocabulary improves dev macro-F1 by about +0.08 (A → C: 0.530 → 0.611) and PERSON F1 from 0.72 to 0.87.

**Compressing the embeddings.** The full embedding matrix does not fit into 16 MB, so it is replaced by a low-rank
factorisation initialised from the pretrained matrix by SVD and exported in int8. Rank 64 (77% of the energy) loses about
0.025 against the uncompressed model (D vs E); ranks 96 (83%) and 128 (88%) are back at the level of the uncompressed
model (0.61-0.63 vs 0.61; differences are within the epoch-to-epoch noise of about ±0.01-0.02 on dev). Rank 96 is the
largest that fits the limit after int8 export (rank 128 is estimated at ≈15 MB for the weights alone).

**Hypothesis 1 (lost capitalisation) - refuted in its original form, with a useful side effect.** The tokenizer is
cased (`Иван` and `иван` get different tokens). On the pruned vocabulary an explicit word-casing embedding still helped
(A → B: +0.027 macro-F1, PERSON 0.72 → 0.81), but on the full vocabulary it adds nothing (C → E: 0.6105 → 0.6134,
within noise). This is consistent with the explanation that the embedding mainly compensated for the casing information
lost in `[UNK]` tokens: once almost nothing maps to `[UNK]`, the benefit disappears. I kept it in the final model because
it costs nothing, but I do not claim it contributes there.

Caveats: single seed; the best epoch is chosen on dev, so the best-epoch numbers are slightly optimistic (the last-epoch
numbers in brackets show the spread); differences below roughly 0.01-0.02 should not be over-interpreted.

### 6. The final model
The final model combines the findings: the whole Cyrillic/Latin vocabulary (83,303 of 83,828 pieces; foreign-script
pieces map to `[UNK]`), a rank-96 factorisation of the embedding matrix initialised by SVD and exported in int8 with
per-row scales, the casing embedding, and a shared encoder with three heads for the nesting layers. About 12.7 MB of
weights; position embeddings are cut to 512; the rest is stored in fp16. Trained on train+dev for 13 epochs (the dev
curve is flat after about 10 epochs).

**Test macro-F1: 0.5994**, with balanced precision and recall (0.603 / 0.609), against 0.5645 for the best CRF.
Dev F1 for a selection of types, CRF vs the transformer (both trained on train only; single runs):

| Type | CRF | Transformer |
|---|---|---|
| PERSON | 0.850 | 0.864 |
| PROFESSION | 0.684 | 0.726 |
| ORGANIZATION | 0.601 | 0.641 |
| COUNTRY | 0.770 | 0.851 |
| DISEASE | 0.317 | 0.638 |
| IDEOLOGY | 0.586 | 0.712 |
| NATIONALITY | 0.485 | 0.667 |
| LAW | 0.414 | 0.534 |
| CRIME | 0.214 | 0.339 |
| LOCATION | 0.429 | 0.400 |
| WORK_OF_ART | 0.663 | 0.435 |

The gains concentrate on rare types and names, as hoped; the transformer is still worse on WORK_OF_ART and LOCATION,
which is a known weak spot and a reason to consider an ensemble.

## Configurations and hyperparameters

### CRF (`train.py`, `solution.py`)
| Parameter | Value |
|---|---|
| Algorithm | L-BFGS, `all_possible_transitions=True` |
| Regularisation | c1 (L1) = 0.05, c2 (L2) = 0.02 (flags `--c1`, `--c2`) |
| Iterations | 150 (`--iters`) |
| Rare feature cut-off | `min_freq` = 2 (`--min-freq`) |
| Nesting layers | 3, one CRF per layer, trained in parallel processes |
| Tokenisation | `razdel` (sentences and words); offsets are mapped back to character positions |
| Model size | 6.05 + 3.09 + 0.84 = 9.99 MB; training about 11 minutes on CPU, prediction on dev 6 s |

Token features: lowercased word, stem (Russian Snowball from `nltk`, works without external data), 2-3 character
prefixes and 1-4 character suffixes, word shape (`Xxxx`, `dd`), casing / uppercase / digits / alphabetic flags, length
(capped at 12), first/last word of the sentence, bigrams with the neighbouring word; for neighbours in a ±2 window:
word, stem, 3-character suffix, shape and casing.

### Transformer (`bert_train.py`, `solution_bert.py`)
| Parameter | Default | Row 6 (first version) | Final (row 7) |
|---|---|---|---|
| Base model | `cointegrated/rubert-tiny2` (3 layers, hidden 312, 12 attention heads) | same | same |
| Data | train | train+dev | train+dev (`--train-on-dev`) |
| Epochs | 10 | 22 | 13 |
| Encoder lr / head lr | 1e-4 / 1e-3 | 3e-4 / 1e-3 | 3e-4 / 1e-3 |
| Batch size | 32 | 16 | 16 |
| Vocabulary | 10,000 frequent tokens + short pieces | 12,000 frequent + short pieces | all Cyrillic/Latin pieces, 83,303 (`--ru-en-vocab`) |
| Token embeddings | full | full | rank-96 factorisation, SVD init, int8 export (`--emb-rank 96 --quantize-emb`) |
| Casing embedding | off | off | on (`--case-emb`) |
| Class weight for `O` | 1.0 | 1.0 | 1.0 |
| Optimiser | AdamW, weight decay 0.01, linear warmup over 6% of steps then linear decay, gradient clipping 1.0 | same | same |
| Other | dropout 0.1 in the heads, seed 42, chunks ≤ 80 words and ≤ 512 positions | same | same |

The loss is the sum of cross-entropies over the three layers, computed on the first sub-token of each word. Training
set size: 7,927 chunks (train) and 9,067 (train+dev). One epoch takes about 9 seconds on a Colab GPU. The exact
parameters of a trained model are stored in `bert_meta.json` (`train_args`).

## Engineering lessons
- The metric and the ceiling of the formulation are computed **before** choosing a model: this immediately showed that
  nesting was the main lever.
- When the transformer lost to the CRF, I did not just tune it: I compared training and dev curves, looked at per-type
  errors, formed hypotheses and wrote a script (`diagnose_vocab.py`) that measures the mechanism without training. One
  hypothesis was confirmed, the other was refuted, and both changed what I built next.
- Every change was checked on dev with the metric formula from the task (macro over types, exact boundary match); only
  verified archives were sent to the grader.
- `check_submission.py` unpacks the archive into a temporary folder and runs it the way the grader does (structure,
  import, output format, timing), catching packaging mistakes before submission.
- Size budgets need explicit accounting: a size estimate that ignored int8 export once suggested a design would not fit
  when it did; the training script now prints the expected size for the chosen flags.
- Practical pitfalls: `crfsuite` on Windows silently fails to save a model into a folder with Cyrillic characters in the
  path; the newer Hugging Face Xet download may be blocked on some networks; character offsets break if text is read
  without `newline=""`.

## Next steps
- Repeat the final configuration with several seeds to quantify the variance of the result.
- An ensemble of the CRF and the transformer, which is better on WORK_OF_ART and LOCATION (needs further compression of
  the encoder, e.g. int8, to fit both into 16 MB).
- Sampling/weighting of rare types for the macro metric; other small Russian encoders.

## Using the released model
The trained model is published in the **Releases** section as `submission_bert.zip` (the exact archive submitted to the
grader). Unpack it and use the `Solution` class (CPU is enough; Python 3.10, `torch`, `transformers`, `tokenizers`,
`razdel`):

```python
from solution import Solution          # solution.py from the unpacked archive

sol = Solution()
texts = ["Президент выступил в Москве на встрече с министрами."]
for text, entities in zip(texts, sol.predict(texts)):
    for start, end, entity_type in sorted(entities):
        print(entity_type, repr(text[start:end]))
```
`predict` returns, for every text, a set of `(start, end, type)` tuples with `[start, end)` character offsets.

## Repository structure
```
README.md / README.ru.md   project description in English and Russian
solution.py                CRF solution (Solution class, features, BIO decoding)
train.py                   NEREL reading, nesting layers, CRF training, metric, dev validation
solution_bert.py           transformer solution (compressed embeddings, nesting layers); submitted as solution.py
bert_train.py              transformer fine-tuning, vocabulary/embedding compression, artifact export
check_submission.py        local check of the archive "as the grader runs it"
diagnose_vocab.py          checks hypotheses without training (casing in the tokenizer, share of [UNK] per vocabulary size)
make_bert_zip.py           builds the submission archive (transformer)
docs/                      method descriptions submitted with the solutions and reproduction instructions
requirements.txt
```
The data (NEREL), model weights and archives are not in the repository (weights are in Releases).

## Reproduction
```bash
pip install -r requirements.txt
# Data: download NEREL v1.1 (https://github.com/nerel-ds/NEREL) and put the train and dev folders
# (*.txt and *.ann files) into data/train and data/dev. Do not use the test folder for training.

# CRF: tune on dev, then train the final model on train+dev
python train.py --data-dir data
python train.py --data-dir data --train-on-dev

# Transformer, final configuration (internet is needed once to download the model; a GPU speeds up training)
python bert_train.py --data-dir data --epochs 13 --lr 3e-4 --batch-size 16 --ru-en-vocab --case-emb --emb-rank 96 --quantize-emb --train-on-dev
# Same flags without --train-on-dev and with --no-save evaluate a configuration on dev only.

# Build the archive and check it locally
python make_bert_zip.py
python check_submission.py submission_bert.zip data/dev
```
For the CRF archive: `python -m zipfile -c submission.zip solution.py model_0.crfsuite model_1.crfsuite model_2.crfsuite description.txt train.py`.
Keep the project path ASCII-only.

## Data and licences
The NEREL corpus is distributed by its authors (https://github.com/nerel-ds/NEREL); see their page for the terms of use.
The pretrained model is `cointegrated/rubert-tiny2` on Hugging Face; the released weights are fine-tuned from it, so
check its licence on the model page before reuse.