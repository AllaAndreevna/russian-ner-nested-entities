import os
import re
from functools import lru_cache
from typing import Iterable, List, Set, Tuple

import sklearn_crfsuite
from nltk.stem.snowball import SnowballStemmer
from razdel import sentenize, tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Сколько "слоёв" вложенности моделируем: слой 0 - самые внешние/длинные сущности,
# слой 1 - вложенные в них, слой 2 - вложенные во вложенные.
NUM_LAYERS = 3

# Порог декодирования: токен получает тег "O", только если P(O) >= O_MIN.
# None - обычный Витерби. Значение подбирается на dev (train.py печатает варианты).
O_MIN = None


def model_path(layer: int) -> str:
    return os.path.join(BASE_DIR, f"model_{layer}.crfsuite")


# ---------- токенизация ----------

def tokenize_text(text: str):
    """Список предложений; предложение = список токенов (start, stop, text)
    с абсолютными смещениями в исходном тексте."""
    sentences = []
    for sent in sentenize(text):
        toks = [
            (sent.start + t.start, sent.start + t.stop, t.text)
            for t in tokenize(sent.text)
        ]
        if toks:
            sentences.append(toks)
    return sentences


# ---------- признаки ----------

_stemmer = SnowballStemmer("russian")


@lru_cache(maxsize=300000)
def stem(lw: str) -> str:
    return _stemmer.stem(lw)


def shape(w: str) -> str:
    s = re.sub(r"[А-ЯЁA-Z]", "X", w)
    s = re.sub(r"[а-яёa-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)  # длинные повторы сжимаем


def token_features(words: List[str], i: int) -> dict:
    w = words[i]
    lw = w.lower()
    f = {
        "bias": 1.0,
        "lw": lw,
        "st": stem(lw),
        "p2": lw[:2],
        "p3": lw[:3],
        "s1": lw[-1:],
        "s2": lw[-2:],
        "s3": lw[-3:],
        "s4": lw[-4:],
        "shape": shape(w),
        "title": w.istitle(),
        "upper": w.isupper(),
        "digit": w.isdigit(),
        "alpha": w.isalpha(),
        "len": min(len(w), 12),
        "first": i == 0,
        "last": i == len(words) - 1,
    }
    if i > 0:
        f["bg-"] = words[i - 1].lower() + "|" + lw
    if i < len(words) - 1:
        f["bg+"] = lw + "|" + words[i + 1].lower()
    for d in (-2, -1, 1, 2):
        j = i + d
        if 0 <= j < len(words):
            v = words[j]
            lv = v.lower()
            f[f"{d}:lw"] = lv
            f[f"{d}:st"] = stem(lv)
            f[f"{d}:s3"] = lv[-3:]
            f[f"{d}:shape"] = shape(v)
            f[f"{d}:title"] = v.istitle()
        else:
            f[f"{d}:pad"] = True
    return f


def sentence_features(words: List[str]) -> List[dict]:
    return [token_features(words, i) for i in range(len(words))]


# ---------- BIO -> сущности ----------

def decode(tokens, tags) -> Set[Tuple[int, int, str]]:
    ents, cur = set(), None
    for (start, stop, _), tag in zip(tokens, tags):
        if tag == "O":
            if cur:
                ents.add(tuple(cur))
                cur = None
            continue
        bio, typ = tag.split("-", 1)
        if bio == "B" or cur is None or cur[2] != typ:
            if cur:
                ents.add(tuple(cur))
            cur = [start, stop, typ]
        else:
            cur[1] = stop
    if cur:
        ents.add(tuple(cur))
    return ents


# ---------- решение ----------

class Solution:
    def __init__(self):
        self.crfs = [
            sklearn_crfsuite.CRF(model_filename=model_path(k)) for k in range(NUM_LAYERS)
        ]
        self.o_min = O_MIN

    def _tags(self, crf, feats):
        if self.o_min is None:
            return crf.predict_single(feats)
        tags = []
        for m in crf.predict_marginals_single(feats):
            if m.get("O", 0.0) >= self.o_min:
                tags.append("O")
            else:
                tags.append(max((l for l in m if l != "O"), key=m.get))
        return tags

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        result = []
        for text in texts:
            ents = set()
            for toks in tokenize_text(text):
                feats = sentence_features([t[2] for t in toks])
                for crf in self.crfs:
                    ents |= decode(toks, self._tags(crf, feats))
            result.append(ents)
        return result