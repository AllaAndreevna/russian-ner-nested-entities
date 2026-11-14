"""Дообучение rubert-tiny2 для NER на NEREL v1.1 (слои вложенности, обрезанный словарь).

Нужен интернет один раз - чтобы скачать cointegrated/rubert-tiny2 с Hugging Face.
Зависимости: pip install torch transformers tokenizers huggingface_hub razdel nltk sklearn-crfsuite

Быстрая проверка:
    python bert_train.py --data-dir data --max-docs 30 --epochs 1
Эксперименты (train -> dev):
    python bert_train.py --data-dir data
Финал (train+dev, число эпох = лучшая эпоха из предыдущего запуска):
    python bert_train.py --data-dir data --train-on-dev --epochs N
Артефакты (bert_model.pt, bert_meta.json, bert_tokenizer.json) сохраняются рядом со скриптом.
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import time
from collections import Counter

import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer, BertConfig, get_linear_schedule_with_warmup

import solution_bert as sb
from train import NUM_LAYERS, doc_to_bio, load_corpus, macro_f1

SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


# ---------- токенизатор и словарь ----------

def fetch_tokenizer_json(model_name, path):
    try:
        from huggingface_hub import hf_hub_download
        shutil.copy(hf_hub_download(model_name, "tokenizer.json"), path)
        return
    except Exception as e:  # noqa: BLE001
        print("tokenizer.json в репозитории не найден, берём из AutoTokenizer:", type(e).__name__)
    tok = AutoTokenizer.from_pretrained(model_name)
    backend = getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer")
    backend.save(path)


def _is_ru_en(t):
    return all(c.isascii() or "а" <= c.lower() <= "я" or c.lower() == "ё" for c in t)


def choose_kept_ids(tok, sentences_words, max_vocab, short_len):
    """Какие токены исходного словаря оставляем: специальные + короткие куски
    (кириллица/латиница/символы) + самые частые токены обучающих текстов."""
    counts = Counter()
    for words in sentences_words:
        counts.update(tok.encode(words, is_pretokenized=True).ids)
    specials = {tok.token_to_id(t) for t in SPECIAL} - {None}
    short = set()
    for token, idx in tok.get_vocab().items():
        t = token[2:] if token.startswith("##") else token
        if 0 < len(t) <= short_len and _is_ru_en(t):
            short.add(idx)
    top = {i for i, _ in counts.most_common(max_vocab)}
    kept = sorted(specials | short | top)
    print(f"токенов в обучающих текстах: {len(counts)}; оставляем {len(kept)} "
          f"(специальных {len(specials)}, коротких {len(short)}, частых {len(top)})")
    return kept


# ---------- данные ----------

def build_samples(docs, encoder, label2id):
    samples = []
    for text, ents in docs:
        for toks, layer_tags in doc_to_bio(text, ents):
            words = [t[2] for t in toks]
            for lo, hi, (ids, first) in encoder.chunks(words):
                Y = [[-100] * len(ids) for _ in range(NUM_LAYERS)]
                for w in range(hi - lo):
                    pos = first[w]
                    if pos is None:
                        continue
                    for k in range(NUM_LAYERS):
                        Y[k][pos] = label2id[layer_tags[k][lo + w]]
                samples.append((ids, Y))
    return samples


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    ids_t = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    Y = torch.full((len(batch), NUM_LAYERS, width), -100, dtype=torch.long)
    for i, (ids, y) in enumerate(batch):
        n = len(ids)
        ids_t[i, :n] = torch.tensor(ids)
        mask[i, :n] = 1
        Y[i, :, :n] = torch.tensor(y)
    return ids_t, mask, Y


def evaluate(model, encoder, labels, dev_docs, verbose=False, title="DEV BERT"):
    texts = [t for t, _ in dev_docs]
    gold = [set(e) for _, e in dev_docs]
    pred = sb.predict_texts(model, encoder, labels, texts)
    return macro_f1(gold, pred, title, verbose=verbose)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-name", default="cointegrated/rubert-tiny2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="lr энкодера")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="lr классификационных голов")
    ap.add_argument("--o-weight", type=float, default=1.0,
                    help="вес класса O в лосе; <1 повышает recall")
    ap.add_argument("--max-vocab", type=int, default=10000,
                    help="сколько самых частых токенов обучающих текстов оставить")
    ap.add_argument("--short-len", type=int, default=2,
                    help="оставлять все куски словаря не длиннее N символов")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--train-on-dev", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("устройство:", device)

    train_docs = load_corpus(os.path.join(args.data_dir, "train"), args.max_docs)
    dev_docs = load_corpus(os.path.join(args.data_dir, "dev"), args.max_docs)
    fit_docs = train_docs + dev_docs if args.train_on_dev else train_docs

    # 1) токенизатор и обрезанный словарь (по тексту обучающей части)
    fetch_tokenizer_json(args.model_name, sb.TOKENIZER_FILE)
    tok = Tokenizer.from_file(sb.TOKENIZER_FILE)
    sent_words = [[t[2] for t in toks] for text, _ in fit_docs for toks in sb.tokenize_text(text)]
    kept = choose_kept_ids(tok, sent_words, args.max_vocab, args.short_len)
    encoder = sb.Encoder(sb.TOKENIZER_FILE, kept)

    # 2) метки
    types = sorted({e[2] for _, ents in fit_docs for e in ents})
    labels = ["O"] + [f"{b}-{t}" for t in types for b in ("B", "I")]
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"типов: {len(types)}, меток: {len(labels)}, слоёв вложенности: {NUM_LAYERS}")

    # 3) модель: предобученный энкодер с урезанными эмбеддингами
    orig = AutoModel.from_pretrained(args.model_name)
    c = orig.config
    bert_cfg = dict(
        vocab_size=len(kept), hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads, intermediate_size=c.intermediate_size,
        hidden_act=c.hidden_act, hidden_dropout_prob=c.hidden_dropout_prob,
        attention_probs_dropout_prob=c.attention_probs_dropout_prob,
        max_position_embeddings=sb.MAX_LEN, type_vocab_size=c.type_vocab_size,
        layer_norm_eps=c.layer_norm_eps, pad_token_id=encoder.pad_id,
    )
    model = sb.NestedBertNER(BertConfig(**bert_cfg), NUM_LAYERS, len(labels))
    sd = {f"bert.{k}": v for k, v in orig.state_dict().items()
          if "pooler" not in k and "position_ids" not in k and "token_type_ids" not in k}
    sd["bert.embeddings.word_embeddings.weight"] = \
        sd["bert.embeddings.word_embeddings.weight"][torch.tensor(kept)]
    sd["bert.embeddings.position_embeddings.weight"] = \
        sd["bert.embeddings.position_embeddings.weight"][: sb.MAX_LEN]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("heads.") and "position_ids" not in k
               and "token_type_ids" not in k]
    print("не загружено из предобученной модели (ожидается пусто):", missing)
    print("лишние ключи (ожидается пусто):", unexpected)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n_params/1e6:.2f} M, в fp16 около {n_params*2/1024/1024:.1f} МБ")
    del orig
    model.to(device)

    # 4) данные
    samples = build_samples(fit_docs, encoder, label2id)
    print(f"обучающих кусков: {len(samples)}")

    # 5) обучение
    opt = torch.optim.AdamW(
        [{"params": model.bert.parameters(), "lr": args.lr},
         {"params": model.heads.parameters(), "lr": args.head_lr}],
        weight_decay=0.01,
    )
    steps = args.epochs * math.ceil(len(samples) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    weight = torch.ones(len(labels))
    weight[0] = args.o_weight
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight.to(device), ignore_index=-100)

    best_score, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(samples)
        t0, total = time.time(), 0.0
        for b in range(0, len(samples), args.batch_size):
            ids, mask, Y = collate(samples[b : b + args.batch_size], encoder.pad_id)
            ids, mask, Y = ids.to(device), mask.to(device), Y.to(device)
            logits = model(ids, mask)  # [K, B, T, L]
            loss = sum(
                loss_fn(logits[k].reshape(-1, len(labels)), Y[:, k].reshape(-1))
                for k in range(NUM_LAYERS)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        msg = f"эпоха {epoch}/{args.epochs}: loss {total/math.ceil(len(samples)/args.batch_size):.3f}, {time.time()-t0:.0f} с"
        if not args.train_on_dev:
            score = evaluate(model, encoder, labels, dev_docs)
            msg += f", dev macro F1 {score:.4f}"
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print(msg)

    # 6) сохранение артефактов (fp16)
    if args.train_on_dev or best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        print(f"\nЛучшая эпоха на dev: {best_epoch} (macro F1 {best_score:.4f}). "
              f"Для финала: --train-on-dev --epochs {best_epoch}")
    state16 = {k: v.half() for k, v in best_state.items() if "position_ids" not in k}
    torch.save(state16, sb.MODEL_FILE)
    with open(sb.META_FILE, "w", encoding="utf-8") as f:
        json.dump({"kept_ids": kept, "labels": labels, "num_layers": NUM_LAYERS,
                   "bert_config": bert_cfg}, f)
    sizes = {os.path.basename(p): os.path.getsize(p) / 1024 / 1024
             for p in (sb.MODEL_FILE, sb.META_FILE, sb.TOKENIZER_FILE)}
    print("размеры файлов, МБ:", {k: round(v, 2) for k, v in sizes.items()},
          f"сумма {sum(sizes.values()):.2f} (лимит на архив 16, сжатие почти не поможет)")

    # 7) проверка "как у проверяющей системы": грузим из файлов и оцениваем
    del model
    sol = sb.Solution()
    suffix = " (dev был в обучении!)" if args.train_on_dev else ""
    t0 = time.time()
    texts = [t for t, _ in dev_docs]
    gold = [set(e) for _, e in dev_docs]
    pred = [set(p) for p in sol.predict(texts)]
    print(f"предсказание на dev из файлов: {time.time()-t0:.0f} с")
    macro_f1(gold, pred, "DEV BERT из сохранённых файлов" + suffix)


if __name__ == "__main__":
    main()
