**English** | [Русский](README.ru.md)

# Named Entity Recognition for Russian News (NEREL): CRF and a Compact Transformer under Hard Constraints

A course project (ISP RAS practicum, autumn 2026). The task is to find named entities of 29 types (PERSON, ORGANIZATION,
PROFESSION, EVENT, DATE, etc.) in Russian news texts, including **nested** entities, and return their boundaries and
types. The solution is submitted to an automated grader with strict limits, so the work is not only about training a
model but also about engineering within a size and time budget.

**Outcome:** the best solution uses three CRF layers for nested entities and reaches **macro-F1 0.5645** on the hidden
test set (up from 0.465 for the first version). A compact transformer (rubert-tiny2 with a pruned vocabulary) reached
0.5346 on the test set and currently trails the CRF; the analysis and next steps are below.

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
| 3 | same, trained on train+dev | same parameters | **0.5645** | 0.659 | 0.509 |
| 4 | rubert-tiny2, 3 heads (one per layer) | train+dev, 9 epochs; everything else default (encoder lr 1e-4, head lr 1e-3, batch 32, vocabulary: 10k most frequent + short pieces) | 0.4956 | 0.557 | 0.484 |
| 5 | rubert-tiny2 | train+dev, 20 epochs; everything else default (as in row 4) | 0.5117 | 0.550 | 0.508 |
| 6 | rubert-tiny2 | train+dev, 22 epochs, encoder lr 3e-4, batch 16, vocabulary: 12k most frequent + short pieces; everything else default | 0.5346 | 0.585 | 0.518 |

Local validation (dev, not used for training) agrees well with the test set: the layered CRF scored 0.560 on dev and
0.547 on test, so dev was a reliable guide for experiments. Prediction time on the grader is 1-2 seconds.

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
Effect on the test set: **+0.08** macro-F1 (0.465 → 0.547). Training on train+dev added another +0.018.

### 3. Error analysis and decoding experiments
In every table precision is noticeably higher than recall, i.e. the model is conservative. I tried biased decoding
(choosing the tag from marginal probabilities with a threshold on the `O` class): on dev it did not help
(0.560 → 0.556 and lower), so it was left out. This negative result is documented to avoid revisiting the idea.

### 4. A compact transformer under the 16 MB limit
To improve rare types, I fine-tuned the pretrained `cointegrated/rubert-tiny2` (3 layers, hidden size 312). The main
engineering problem is size: the model's vocabulary (83,828 tokens) accounts for most of its weight. The solution:
- the vocabulary is pruned to special tokens, short pieces (≤ 2 characters) and the most frequent tokens of the
  training texts (about 12k); all other tokens map to `[UNK]`; position embeddings are cut to 512;
- weights are stored in fp16: **6.4M parameters, about 12 MB** (archive about 12 MB against the 16 MB limit);
- nested entities use the same trick: a shared encoder with three classification heads, one per layer;
- pure PyTorch inference with limited threads, 3 seconds for the 89 dev texts;
- epochs, learning rate and vocabulary size were tuned on dev in Google Colab.

### 5. Why the transformer fell short
On dev (without training on it) the transformer plateaued around 0.52 after about 10 epochs while the training loss
kept decreasing, so the problem is generalisation rather than underfitting. By type it loses to the CRF almost
everywhere, especially on PERSON (F1 0.71 vs 0.85), that is, on names, which are rare words.

Hypotheses (**not yet verified experimentally**):
1. The model's tokenizer lowercases the text and loses the capital letter, a strong cue for proper names that the CRF
   gets explicitly.
2. Frequency-based vocabulary pruning maps exactly the rare tokens (names, titles) to `[UNK]` and cancels the benefit of
   pretraining for unseen words.

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
| Parameter | Default | Best run (row 6) |
|---|---|---|
| Base model | `cointegrated/rubert-tiny2` (3 layers, hidden 312, 12 attention heads) | same |
| Data | train | train+dev (`--train-on-dev`) |
| Epochs | 10 | 22 |
| Encoder lr / head lr | 1e-4 / 1e-3 | 3e-4 / 1e-3 |
| Batch size | 32 | 16 |
| Vocabulary size | 10,000 frequent tokens + short pieces | 12,000 frequent + short pieces |
| Class weight for `O` | 1.0 | 1.0 |
| Optimiser | AdamW, weight decay 0.01, linear warmup over 6% of steps then linear decay, gradient clipping 1.0 | same |
| Other | dropout 0.1 in the heads, seed 42, chunks ≤ 80 words and ≤ 512 positions | same |

The loss is the sum of cross-entropies over the three layers, computed on the first sub-token of each word. Weights are
saved in fp16. Training set size: 7,927 chunks (train) and 9,067 (train+dev). One epoch takes roughly 4 minutes on CPU.
Rows 4 and 5 of the results table use the default parameters and differ only in the number of epochs.

## Engineering lessons
- The metric and the ceiling of the formulation are computed **before** choosing a model: this immediately showed that
  nesting was the main lever.
- Every change was checked on dev with the metric formula from the task (macro over types, exact boundary match); only
  verified archives were sent to the grader.
- `check_submission.py` unpacks the archive into a temporary folder and runs it the way the grader does (structure,
  import, output format, timing), catching packaging mistakes before submission.
- Practical pitfalls: `crfsuite` on Windows silently fails to save a model into a folder with Cyrillic characters in the
  path; the newer Hugging Face Xet download may be blocked on some networks; character offsets break if text is read
  without `newline=""`.

## Next steps
- Add a separate word-casing embedding to the transformer and test hypothesis 1.
- Keep the full needed vocabulary but store embeddings compactly (low-rank factorisation / int8) and test hypothesis 2.
- A CRF + transformer ensemble (currently does not fit in 16 MB; needs quantisation and stronger CRF regularisation).
- Sampling/weighting of rare types for the macro metric.

## Repository structure
```
README.md / README.ru.md   project description in English and Russian
solution.py                CRF solution (Solution class, features, BIO decoding)
train.py                   NEREL reading, nesting layers, CRF training, metric, dev validation
solution_bert.py           rubert-tiny2 solution (pruned vocabulary, nesting layers)
bert_train.py              transformer fine-tuning, vocabulary pruning, artifact export
check_submission.py        local check of the archive "as the grader runs it"
make_bert_zip.py           builds the submission archive (transformer)
docs/                      method descriptions submitted with the solutions and reproduction instructions
requirements.txt
```
The data (NEREL), model weights and archives are not included in the repository.

## Reproduction
```bash
pip install -r requirements.txt
# Data: download NEREL v1.1 (https://github.com/nerel-ds/NEREL) and put the train and dev folders
# (*.txt and *.ann files) into data/train and data/dev. Do not use the test folder for training.

# CRF: tune on dev, then train the final model on train+dev
python train.py --data-dir data
python train.py --data-dir data --train-on-dev

# Transformer (internet is needed once to download the model; a GPU speeds up training; best-run parameters)
python bert_train.py --data-dir data --train-on-dev --epochs 22 --lr 3e-4 --batch-size 16 --max-vocab 12000

# Build the archive and check it locally
python make_bert_zip.py
python check_submission.py submission_bert.zip data/dev
```
For the CRF archive: `python -m zipfile -c submission.zip solution.py model_0.crfsuite model_1.crfsuite model_2.crfsuite description.txt train.py`.
Keep the project path ASCII-only.

## Data and licences
The NEREL corpus is distributed by its authors (https://github.com/nerel-ds/NEREL/tree/master/NEREL-v1.1); see their page for the terms of use.
The pretrained model is `cointegrated/rubert-tiny2` on Hugging Face; check the licence on the model page.