"""Обучение слоёв CRF для NER на NEREL v1.1 и валидация на dev.

Запуск (из папки с solution.py, путь к папке только латиницей):
    python train.py --data-dir data
Быстрая проверка:
    python train.py --data-dir data --max-docs 30 --iters 20
Финальная модель (train + dev, параметры уже выбраны):
    python train.py --data-dir data --train-on-dev
"""
import argparse
import glob
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import sklearn_crfsuite

from solution import (
    NUM_LAYERS,
    Solution,
    decode,
    model_path,
    sentence_features,
    tokenize_text,
)


# ---------- чтение корпуса ----------

def read_ann(path):
    """Возвращает (сущности {(start, end, type)}, есть_ли_разрывные)."""
    ents, has_disc = set(), False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("T"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            info = parts[1]
            if ";" in info:  # разрывная сущность
                has_disc = True
                continue
            typ, s, e = info.split()
            ents.add((int(s), int(e), typ))
    return ents, has_disc


def load_corpus(directory, max_docs=None):
    """Тексты с разрывными сущностями пропускаем (как в тесте)."""
    docs, skipped = [], 0
    for txt_path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        ann_path = txt_path[:-4] + ".ann"
        if not os.path.exists(ann_path):
            continue
        # newline="" важно: иначе \r\n превратится в \n и смещения поедут
        with open(txt_path, encoding="utf-8", newline="") as f:
            text = f.read()
        ents, has_disc = read_ann(ann_path)
        if has_disc:
            skipped += 1
            continue
        docs.append((text, ents))
        if max_docs and len(docs) >= max_docs:
            break
    print(f"{directory}: {len(docs)} текстов (пропущено с разрывными: {skipped})")
    return docs


# ---------- сущности -> слои -> BIO ----------

def assign_layers(ents, num_layers):
    """Раскладываем сущности по слоям так, чтобы внутри слоя спаны не пересекались.
    Сначала длинные: слой 0 - внешние, дальше - вложенные. Не влезшие отбрасываем."""
    layers = [[] for _ in range(num_layers)]
    for s, e, typ in sorted(ents, key=lambda x: (-(x[1] - x[0]), x[0], x[2])):
        for layer in layers:
            if all(e <= ls or s >= le for ls, le, _ in layer):
                layer.append((s, e, typ))
                break
    return layers


def to_bio(tokens, ents):
    """Плоская BIO-разметка одного слоя (внутри слоя пересечений уже нет)."""
    lo, hi = tokens[0][0], tokens[-1][1]
    tags = ["O"] * len(tokens)
    for s, e, typ in ents:
        if not (s < hi and e > lo):
            continue
        idx = [i for i, (ts, te, _) in enumerate(tokens) if ts < e and te > s]
        if not idx or any(tags[i] != "O" for i in idx):
            continue
        tags[idx[0]] = "B-" + typ
        for i in idx[1:]:
            tags[i] = "I-" + typ
    return tags


def doc_to_bio(text, ents):
    """[(токены предложения, [теги слоя 0, теги слоя 1, ...])]"""
    layers = assign_layers(ents, NUM_LAYERS)
    return [
        (toks, [to_bio(toks, layer) for layer in layers])
        for toks in tokenize_text(text)
    ]


def build_dataset(docs):
    X, ys = [], [[] for _ in range(NUM_LAYERS)]
    for text, ents in docs:
        for toks, layer_tags in doc_to_bio(text, ents):
            X.append(sentence_features([t[2] for t in toks]))
            for k in range(NUM_LAYERS):
                ys[k].append(layer_tags[k])
    return X, ys


# ---------- метрика из условия ----------

def macro_f1(gold, pred, title="", verbose=True):
    """Среднее по типам F1_t, где F1_t = 2PR/(P+R) по множествам сущностей типа t."""
    types = sorted({e[2] for g in gold for e in g})
    n_gold, n_pred, tp = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold, pred):
        for e in g:
            n_gold[e[2]] += 1
        for e in p:
            n_pred[e[2]] += 1
        for e in g & p:
            tp[e[2]] += 1
    rows, scores = [], []
    for t in types:
        P = tp[t] / n_pred[t] if n_pred[t] else 0.0
        R = tp[t] / n_gold[t] if n_gold[t] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        rows.append((n_gold[t], t, P, R, F))
        scores.append(F)
    score = sum(scores) / len(scores) if scores else 0.0
    if verbose:
        print(f"\n=== {title} ===")
        print(f"{'тип':28s}{'gold':>7s}{'P':>8s}{'R':>8s}{'F1':>8s}")
        for n, t, P, R, F in sorted(rows, reverse=True):
            print(f"{t:28s}{n:7d}{P:8.3f}{R:8.3f}{F:8.3f}")
    print(f"MACRO F1 по типам ({title}): {score:.4f}")
    return score


# ---------- обучение одного слоя (запускается в отдельном процессе) ----------

def fit_layer(job):
    k, X, y, params = job
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=params["c1"],
        c2=params["c2"],
        max_iterations=params["iters"],
        min_freq=params["min_freq"],
        all_possible_transitions=True,
        model_filename=model_path(k),
        verbose=False,
    )
    t0 = time.time()
    crf.fit(X, y)
    return k, time.time() - t0, os.path.getsize(model_path(k)) / 1024 / 1024


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="папка с train/ и dev/")
    ap.add_argument("--c1", type=float, default=0.05, help="L1: больше -> меньше модель")
    ap.add_argument("--c2", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--min-freq", type=int, default=2, help="отбросить редкие признаки")
    ap.add_argument("--max-docs", type=int, default=None, help="для быстрых тестов")
    ap.add_argument("--train-on-dev", action="store_true", help="финал: train+dev")
    args = ap.parse_args()

    train_docs = load_corpus(os.path.join(args.data_dir, "train"), args.max_docs)
    dev_docs = load_corpus(os.path.join(args.data_dir, "dev"), args.max_docs)
    fit_docs = train_docs + dev_docs if args.train_on_dev else train_docs

    t0 = time.time()
    X, ys = build_dataset(fit_docs)
    n_ent = [sum(1 for tg in y for t in tg if t.startswith("B-")) for y in ys]
    print(f"предложений: {len(X)}, признаки за {time.time()-t0:.0f} с; "
          f"сущностей по слоям: {n_ent}")

    params = dict(c1=args.c1, c2=args.c2, iters=args.iters, min_freq=args.min_freq)
    jobs = [(k, X, ys[k], params) for k in range(NUM_LAYERS)]
    print(f"обучаем {NUM_LAYERS} слоя параллельно (логов не будет, ждите)...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(NUM_LAYERS, os.cpu_count() or 1)) as ex:
        results = list(ex.map(fit_layer, jobs))
    total = 0.0
    for k, sec, size in sorted(results):
        print(f"слой {k}: {sec:.0f} с, модель {size:.2f} МБ")
        total += size
    print(f"всего {time.time()-t0:.0f} с, суммарный размер моделей {total:.2f} МБ "
          f"(лимит на весь архив 16 МБ)")

    texts = [t for t, _ in dev_docs]
    gold = [set(e) for _, e in dev_docs]
    sol = Solution()
    suffix = " (dev был в обучении!)" if args.train_on_dev else ""

    # 1) обычный Витерби
    sol.o_min = None
    t0 = time.time()
    pred = [set(p) for p in sol.predict(texts)]
    print(f"предсказание на dev: {time.time()-t0:.0f} с")
    macro_f1(gold, pred, "DEV Витерби" + suffix)

    # 2) подбор порога O_MIN (больше -> больше сущностей -> выше recall)
    best = (macro_f1(gold, pred, "o_min=None", verbose=False), None)
    for thr in (0.5, 0.65, 0.8, 0.9):
        sol.o_min = thr
        p = [set(x) for x in sol.predict(texts)]
        score = macro_f1(gold, p, f"o_min={thr}", verbose=False)
        if score > best[0]:
            best = (score, thr)
    print(f"\nЛучший вариант на dev: o_min={best[1]} (macro F1 {best[0]:.4f}). "
          f"Впишите его в solution.py: O_MIN = {best[1]}")

    # потолок: gold -> слои -> BIO -> сущности
    oracle = []
    for text, ents in dev_docs:
        s = set()
        for toks, layer_tags in doc_to_bio(text, ents):
            for tags in layer_tags:
                s |= decode(toks, tags)
        oracle.append(s)
    macro_f1(gold, oracle, "DEV потолок разметки по слоям, без модели")


if __name__ == "__main__":
    main()