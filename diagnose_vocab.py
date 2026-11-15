"""Проверка гипотез о трансформере БЕЗ обучения.

Гипотеза 1: токенизатор rubert-tiny2 теряет регистр (Иван == иван).
Гипотеза 2: обрезка словаря превращает в [UNK] именно куски сущностей.

    python diagnose_vocab.py --data-dir data
"""
import argparse
import os
from collections import defaultdict

from tokenizers import Tokenizer

import solution_bert as sb
from bert_train import choose_kept_ids, fetch_tokenizer_json
from train import load_corpus

SHOW_TYPES = ("PERSON", "ORGANIZATION", "PROFESSION", "EVENT", "COUNTRY")


def piece_ids_per_word(tok, words):
    enc = tok.encode(words, is_pretokenized=True)
    per_word = [[] for _ in words]
    for pid, w in zip(enc.ids, enc.word_ids):
        if w is not None:
            per_word[w].append(pid)
    return per_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-name", default="cointegrated/rubert-tiny2")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(sb.TOKENIZER_FILE):
        fetch_tokenizer_json(args.model_name, sb.TOKENIZER_FILE)
    tok = Tokenizer.from_file(sb.TOKENIZER_FILE)

    # ---- гипотеза 1 ----
    print("=== Гипотеза 1: регистр в токенизаторе ===")
    same_all = True
    for a, b in [("Иван", "иван"), ("ИВАН", "иван"), ("Москва", "москва"), ("США", "сша")]:
        ea, eb = tok.encode(a), tok.encode(b)
        same = ea.ids == eb.ids
        same_all &= same
        print(f"{a!r} -> {ea.tokens[1:-1]} | {b!r} -> {eb.tokens[1:-1]} | одинаковые id: {same}")
    if same_all:
        print("ВЫВОД: токенизатор приводит текст к нижнему регистру - заглавная буква теряется (гипотеза 1 подтверждена "
              "на уровне токенизатора; влияние на F1 проверяет эксперимент с --case-emb).")
    else:
        print("ВЫВОД: токенизатор различает регистр хотя бы частично - гипотеза 1 в исходном виде неверна.")

    # ---- гипотеза 2 ----
    print("\n=== Гипотеза 2: [UNK] при обрезке словаря (словарь строится по train, измеряется на dev) ===")
    train_docs = load_corpus(os.path.join(args.data_dir, "train"), args.max_docs)
    dev_docs = load_corpus(os.path.join(args.data_dir, "dev"), args.max_docs)
    sent_words = [[t[2] for t in toks] for text, _ in train_docs for toks in sb.tokenize_text(text)]

    dev_data = []  # на документ: ([(start, stop, [id кусков])], сущности)
    for text, ents in dev_docs:
        words = []
        for toks in sb.tokenize_text(text):
            per_word = piece_ids_per_word(tok, [t[2] for t in toks])
            for (s, e, _), pids in zip(toks, per_word):
                words.append((s, e, pids))
        dev_data.append((words, ents))

    configs = [
        ("10 000 частых", dict(max_vocab=10000)),
        ("12 000 частых", dict(max_vocab=12000)),
        ("20 000 частых", dict(max_vocab=20000)),
        ("все токены train", dict(max_vocab=10 ** 9)),
        ("весь словарь", dict(full_vocab=True)),
    ]
    rows = []
    for name, kw in configs:
        kept = set(choose_kept_ids(tok, sent_words, kw.get("max_vocab", 0), 2, kw.get("full_vocab", False)))
        n_pieces = n_unk = 0
        tot, unk = defaultdict(int), defaultdict(int)
        for words, ents in dev_data:
            word_unk = []
            for _, _, pids in words:
                n_pieces += len(pids)
                bad = sum(p not in kept for p in pids)
                n_unk += bad
                word_unk.append(bad > 0)
            for s, e, typ in ents:
                hit = any(word_unk[i] for i, (ws, we, _) in enumerate(words) if ws < e and we > s)
                tot[typ] += 1
                unk[typ] += hit
        n_ent, n_ent_unk = sum(tot.values()), sum(unk.values())
        rows.append((name, len(kept), 100 * n_unk / max(n_pieces, 1),
                     100 * n_ent_unk / max(n_ent, 1),
                     {t: 100 * unk[t] / max(tot[t], 1) for t in SHOW_TYPES}))

    print("\nДоля [UNK] на dev (в процентах):")
    header = f"{'словарь':20s}{'токенов':>9s}{'кусков':>9s}{'сущностей':>11s}" + "".join(f"{t[:6]:>9s}" for t in SHOW_TYPES)
    print(header)
    for name, size, piece_pct, ent_pct, by_type in rows:
        print(f"{name:20s}{size:9d}{piece_pct:9.1f}{ent_pct:11.1f}" + "".join(f"{by_type[t]:9.1f}" for t in SHOW_TYPES))
    print("\n'кусков' - доля кусков слов, ставших [UNK]; 'сущностей' - доля эталонных сущностей dev, "
          "содержащих хотя бы один [UNK]; дальше - то же по типам.")


if __name__ == "__main__":
    main()
