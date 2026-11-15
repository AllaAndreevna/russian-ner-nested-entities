"""Дообучение rubert-tiny2 для NER на NEREL v1.1 (слои вложенности, сжатый словарь).

Нужен интернет один раз - чтобы скачать cointegrated/rubert-tiny2 с Hugging Face.
Зависимости: pip install torch transformers tokenizers huggingface_hub razdel nltk sklearn-crfsuite

Эксперимент на dev (результаты не сохраняются, печатается таблица по типам):
    python bert_train.py --data-dir data --epochs 15 --lr 3e-4 --batch-size 16 --max-vocab 12000 --no-save
Гипотеза 1 (регистр):   добавить --case-emb
Гипотеза 2 (словарь):   заменить --max-vocab 12000 на --full-vocab
Всё вместе и в 16 МБ:   --full-vocab --case-emb --emb-rank 64 --quantize-emb
Финал (train+dev, число эпох = лучшая эпоха из эксперимента):
    python bert_train.py --data-dir data --train-on-dev --epochs N <те же флаги>
Артефакты (bert_model.pt, bert_meta.json, bert_tokenizer.json) сохраняются рядом со скриптом.
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter

# Обычная HTTP-загрузка вместо Xet (Xet-сервер у многих недоступен) - ДО импорта transformers
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer, BertConfig, get_linear_schedule_with_warmup

import solution_bert as sb
from train import NUM_LAYERS, doc_to_bio, load_corpus, macro_f1

SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


# ---------- токенизатор и словарь ----------

def fetch_tokenizer_json(model_name, path):
    local = os.path.join(model_name, "tokenizer.json")
    if os.path.isdir(model_name) and os.path.exists(local):  # модель скачана вручную
        shutil.copy(local, path)
        return
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


def choose_kept_ids(tok, sentences_words, max_vocab, short_len, full_vocab=False, ru_en_only=False):
    """Какие токены исходного словаря оставляем: специальные + короткие куски
    (кириллица/латиница/символы) + самые частые токены обучающих текстов.
    full_vocab=True - оставить весь словарь;
    ru_en_only=True - весь словарь, но только кириллица/латиница/цифры/знаки (+ специальные токены)."""
    if full_vocab or ru_en_only:
        vocab = tok.get_vocab()
        if ru_en_only:
            specials = {tok.token_to_id(t) for t in SPECIAL} - {None}
            ids = specials | {idx for token, idx in vocab.items()
                              if _is_ru_en(token[2:] if token.startswith("##") else token)}
        else:
            ids = set(vocab.values())
        kept = sorted(ids)
        print(f"словарь {'ограничен кириллицей/латиницей' if ru_en_only else 'не обрезается'}: "
              f"{len(kept)} токенов из {len(vocab)}")
        return kept
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
            for lo, hi, (ids, first, case) in encoder.chunks(words):
                Y = [[-100] * len(ids) for _ in range(NUM_LAYERS)]
                for w in range(hi - lo):
                    pos = first[w]
                    if pos is None:
                        continue
                    for k in range(NUM_LAYERS):
                        Y[k][pos] = label2id[layer_tags[k][lo + w]]
                samples.append((ids, Y, case))
    return samples


def collate(batch, pad_id):
    width = max(len(ids) for ids, _, _ in batch)
    ids_t = torch.full((len(batch), width), pad_id, dtype=torch.long)
    case_t = torch.zeros((len(batch), width), dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    Y = torch.full((len(batch), NUM_LAYERS, width), -100, dtype=torch.long)
    for i, (ids, y, case) in enumerate(batch):
        n = len(ids)
        ids_t[i, :n] = torch.tensor(ids)
        case_t[i, :n] = torch.tensor(case)
        mask[i, :n] = 1
        Y[i, :, :n] = torch.tensor(y)
    return ids_t, mask, case_t, Y


def evaluate(model, encoder, labels, docs, verbose=False, title="DEV BERT"):
    texts = [t for t, _ in docs]
    gold = [set(e) for _, e in docs]
    pred = sb.predict_texts(model, encoder, labels, texts)
    return macro_f1(gold, pred, title, verbose=verbose)


# ---------- эмбеддинги и экспорт ----------

def init_token_embeddings(tok_emb, W, rank):
    """Инициализация эмбеддингов токенов из предобученной матрицы W [V, H]."""
    with torch.no_grad():
        if not rank:
            tok_emb.E.weight.copy_(W)
            return
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        tok_emb.A.weight.copy_(U[:, :rank] * S[:rank])
        tok_emb.B.weight.copy_(Vh[:rank].T)
        energy = (S[:rank] ** 2).sum() / (S ** 2).sum()
        print(f"низкоранговое разложение эмбеддингов: ранг {rank}, сохранено {energy * 100:.1f}% энергии")


def export_state(state, quantize):
    """fp16-веса; матрица эмбеддингов токенов по желанию - int8 с масштабом по строкам."""
    out = {}
    for k, v in state.items():
        if quantize and k in ("tok_emb.A.weight", "tok_emb.E.weight"):
            scale = (v.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
            out[k + "__int8"] = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
            out[k + "__scale"] = scale.half()
        else:
            out[k] = v.half() if v.is_floating_point() else v
    return out


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
    ap.add_argument("--full-vocab", action="store_true",
                    help="не обрезать словарь (диагностика или вместе с --emb-rank)")
    ap.add_argument("--ru-en-vocab", action="store_true",
                    help="весь словарь, но только кириллица/латиница/цифры/знаки (меньше строк в матрице эмбеддингов)")
    ap.add_argument("--emb-rank", type=int, default=0,
                    help="ранг низкорангового разложения эмбеддингов токенов (0 - без разложения)")
    ap.add_argument("--quantize-emb", action="store_true",
                    help="хранить матрицу эмбеддингов токенов в int8")
    ap.add_argument("--case-emb", action="store_true",
                    help="добавить эмбеддинг регистра слова (гипотеза 1)")
    ap.add_argument("--no-save", action="store_true",
                    help="эксперимент: не сохранять артефакты")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--train-on-dev", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(vars(args))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("устройство:", device)

    train_docs = load_corpus(os.path.join(args.data_dir, "train"), args.max_docs)
    dev_docs = load_corpus(os.path.join(args.data_dir, "dev"), args.max_docs)
    fit_docs = train_docs + dev_docs if args.train_on_dev else train_docs

    # 1) токенизатор и словарь (по тексту обучающей части)
    fetch_tokenizer_json(args.model_name, sb.TOKENIZER_FILE)
    tok = Tokenizer.from_file(sb.TOKENIZER_FILE)
    sent_words = [[t[2] for t in toks] for text, _ in fit_docs for toks in sb.tokenize_text(text)]
    kept = choose_kept_ids(tok, sent_words, args.max_vocab, args.short_len, args.full_vocab, args.ru_en_vocab)
    encoder = sb.Encoder(sb.TOKENIZER_FILE, kept)

    # 2) метки
    types = sorted({e[2] for _, ents in fit_docs for e in ents})
    labels = ["O"] + [f"{b}-{t}" for t in types for b in ("B", "I")]
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"типов: {len(types)}, меток: {len(labels)}, слоёв вложенности: {NUM_LAYERS}")

    # 3) модель: предобученный энкодер, эмбеддинги токенов из предобученной матрицы
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
    model = sb.NestedBertNER(
        BertConfig(**bert_cfg), NUM_LAYERS, len(labels),
        emb_rank=args.emb_rank, use_case=args.case_emb,
    )
    orig_sd = orig.state_dict()
    init_token_embeddings(
        model.tok_emb, orig_sd["embeddings.word_embeddings.weight"][torch.tensor(kept)], args.emb_rank
    )
    sd = {f"bert.{k}": v for k, v in orig_sd.items()
          if "pooler" not in k and "position_ids" not in k and "token_type_ids" not in k
          and "word_embeddings" not in k}
    sd["bert.embeddings.position_embeddings.weight"] = \
        sd["bert.embeddings.position_embeddings.weight"][: sb.MAX_LEN]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing
               if not k.startswith(("heads.", "tok_emb.", "case_emb."))
               and "position_ids" not in k and "token_type_ids" not in k
               and "word_embeddings" not in k]
    print("не загружено из предобученной модели (ожидается пусто):", missing)
    print("лишние ключи (ожидается пусто):", unexpected)
    n_params = sum(p.numel() for p in model.parameters())
    emb_w = model.tok_emb.A.weight if args.emb_rank else model.tok_emb.E.weight
    emb_n = emb_w.numel()
    est_fp16 = n_params * 2
    est_export = (n_params - emb_n) * 2 + (emb_n + 2 * emb_w.shape[0] if args.quantize_emb else emb_n * 2)
    print(f"параметров: {n_params/1e6:.2f} M; размер весов: fp16 около {est_fp16/1024/1024:.1f} МБ, "
          f"при экспорте с текущими флагами (--quantize-emb={args.quantize_emb}) около {est_export/1024/1024:.1f} МБ "
          f"(+ токенизатор и meta ~2 МБ до сжатия архива)")
    del orig, orig_sd
    model.to(device)

    # 4) данные
    samples = build_samples(fit_docs, encoder, label2id)
    print(f"обучающих кусков: {len(samples)}")

    # 5) обучение
    heads_params = list(model.heads.parameters())
    enc_params = [p for n, p in model.named_parameters() if not n.startswith("heads.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr}, {"params": heads_params, "lr": args.head_lr}],
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
            ids, mask, case, Y = collate(samples[b : b + args.batch_size], encoder.pad_id)
            ids, mask, case, Y = ids.to(device), mask.to(device), case.to(device), Y.to(device)
            logits = model(ids, mask, case)  # [K, B, T, L]
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
            tr_score = evaluate(model, encoder, labels, train_docs[:100], title="TRAIN100")
            msg += f", train100 F1 {tr_score:.4f}"
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print(msg)

    # 6) лучшее состояние обратно в модель + итоговая таблица по типам
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nЛучшая эпоха на dev: {best_epoch} (macro F1 {best_score:.4f}). "
              f"Для финала: --train-on-dev --epochs {best_epoch}")
        evaluate(model, encoder, labels, dev_docs, verbose=True, title="DEV BERT, лучшая эпоха")
    if args.no_save:
        print("--no-save: артефакты не сохраняем")
        return

    # 7) сохранение артефактов
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "position_ids" not in k}
    torch.save(export_state(state, args.quantize_emb), sb.MODEL_FILE)
    with open(sb.META_FILE, "w", encoding="utf-8") as f:
        json.dump({"kept_ids": kept, "labels": labels, "num_layers": NUM_LAYERS,
                   "bert_config": bert_cfg, "emb_rank": args.emb_rank,
                   "use_case": args.case_emb, "train_args": vars(args)}, f)
    sizes = {os.path.basename(p): os.path.getsize(p) / 1024 / 1024
             for p in (sb.MODEL_FILE, sb.META_FILE, sb.TOKENIZER_FILE)}
    total_mb = sum(sizes.values())
    print("размеры файлов, МБ:", {k: round(v, 2) for k, v in sizes.items()},
          f"сумма {total_mb:.2f} (лимит на архив 16, сжатие почти не поможет)")
    print("Команда этого запуска (для README): python bert_train.py " + " ".join(sys.argv[1:]))
    if total_mb > 15:
        print("ВНИМАНИЕ: близко к лимиту 16 МБ или выше - уменьшите --emb-rank / --max-vocab")

    # 8) проверка "как у проверяющей системы": грузим из файлов и оцениваем
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