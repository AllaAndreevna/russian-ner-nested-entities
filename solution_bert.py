"""NER на основе дообученного rubert-tiny2 (обрезанный словарь, слои вложенности).

В архиве для проверяющей системы этот файл кладётся как solution.py
(см. make_bert_zip.py). Артефакты обучения лежат рядом с ним:
bert_model.pt, bert_meta.json, bert_tokenizer.json.
"""
import json
import os
from typing import Iterable, List, Set, Tuple

import numpy as np
import torch
from razdel import sentenize, tokenize
from tokenizers import Tokenizer
from transformers import BertConfig, BertModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "bert_model.pt")
META_FILE = os.path.join(BASE_DIR, "bert_meta.json")
TOKENIZER_FILE = os.path.join(BASE_DIR, "bert_tokenizer.json")

MAX_LEN = 512      # позиций у обрезанной модели (вместе с [CLS] и [SEP])
MAX_WORDS = 80     # слов в одном куске предложения
NUM_THREADS = 4    # ограничение потоков для torch


# ---------- токенизация и декодирование (те же, что в CRF-версии) ----------

def tokenize_text(text: str):
    """Список предложений; предложение = список токенов (start, stop, text)."""
    sentences = []
    for sent in sentenize(text):
        toks = [
            (sent.start + t.start, sent.start + t.stop, t.text)
            for t in tokenize(sent.text)
        ]
        if toks:
            sentences.append(toks)
    return sentences


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


# ---------- кодирование слов в id обрезанного словаря ----------

class Encoder:
    """Токенизатор исходной модели + отображение старых id в компактные.
    Токены, которых нет среди kept_ids, превращаются в [UNK]."""

    def __init__(self, tokenizer_path, kept_ids):
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.no_truncation()
        self.tok.no_padding()
        self.kept_ids = list(kept_ids)
        new_of_old = {old: new for new, old in enumerate(self.kept_ids)}
        self.pad_id = new_of_old[self.tok.token_to_id("[PAD]")]
        self.unk_id = new_of_old[self.tok.token_to_id("[UNK]")]
        size = self.tok.get_vocab_size()
        self.id_map = np.full(size, self.unk_id, dtype=np.int64)
        for old, new in new_of_old.items():
            self.id_map[old] = new

    def encode(self, words):
        """-> (ids с [CLS]/[SEP], позиция первого подтокена каждого слова или None)."""
        enc = self.tok.encode(words, is_pretokenized=True)
        ids = self.id_map[np.asarray(enc.ids, dtype=np.int64)].tolist()
        first = [None] * len(words)
        for pos, w in enumerate(enc.word_ids):
            if w is not None and first[w] is None:
                first[w] = pos
        if len(ids) > MAX_LEN:  # страховка от патологически длинных слов
            ids = ids[: MAX_LEN - 1] + ids[-1:]
            first = [p if p is not None and p < MAX_LEN - 1 else None for p in first]
        return ids, first

    def chunks(self, words):
        """Режет предложение на куски, влезающие в MAX_LEN: [(lo, hi, (ids, first))]."""
        out = []

        def rec(lo, hi):
            enc = self.encode(words[lo:hi])
            if len(enc[0]) <= MAX_LEN or hi - lo == 1:
                out.append((lo, hi, enc))
                return
            mid = (lo + hi) // 2
            rec(lo, mid)
            rec(mid, hi)

        for lo in range(0, len(words), MAX_WORDS):
            rec(lo, min(lo + MAX_WORDS, len(words)))
        return out


# ---------- модель ----------

class NestedBertNER(torch.nn.Module):
    """Общий энкодер и по одной классификационной "голове" на каждый слой вложенности."""

    def __init__(self, config, n_layers, n_labels):
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=False)
        self.drop = torch.nn.Dropout(0.1)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(config.hidden_size, n_labels) for _ in range(n_layers)]
        )

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.drop(h)
        return torch.stack([head(h) for head in self.heads], dim=0)  # [K, B, T, L]


@torch.inference_mode()
def predict_texts(model, encoder, labels, texts, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(model.heads)
    sents, tags_store, items = [], [], []
    for ti, text in enumerate(texts):
        for toks in tokenize_text(text):
            sid = len(sents)
            sents.append((ti, toks))
            tags_store.append([["O"] * len(toks) for _ in range(n_layers)])
            for lo, hi, (ids, first) in encoder.chunks([t[2] for t in toks]):
                items.append((sid, lo, hi, ids, first))

    order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
    for b in range(0, len(order), batch_size):
        batch = [items[i] for i in order[b : b + batch_size]]
        width = max(len(it[3]) for it in batch)
        ids = torch.full((len(batch), width), encoder.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for bi, it in enumerate(batch):
            n = len(it[3])
            ids[bi, :n] = torch.tensor(it[3], dtype=torch.long)
            mask[bi, :n] = 1
        pred = model(ids.to(device), mask.to(device)).argmax(-1).cpu().tolist()  # [K][B][T]
        for bi, (sid, lo, hi, _, first) in enumerate(batch):
            for w in range(hi - lo):
                pos = first[w]
                if pos is None:
                    continue
                for k in range(n_layers):
                    tags_store[sid][k][lo + w] = labels[pred[k][bi][pos]]

    result = [set() for _ in texts]
    for (ti, toks), layer_tags in zip(sents, tags_store):
        for tags in layer_tags:
            result[ti] |= decode(toks, tags)
    return result


# ---------- решение ----------

class Solution:
    def __init__(self):
        torch.set_num_threads(NUM_THREADS)
        with open(META_FILE, encoding="utf-8") as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.encoder = Encoder(TOKENIZER_FILE, meta["kept_ids"])
        self.model = NestedBertNER(
            BertConfig(**meta["bert_config"]), meta["num_layers"], len(self.labels)
        )
        state = torch.load(MODEL_FILE, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(
            {k: v.float() for k, v in state.items()}, strict=False
        )
        bad = [k for k in list(missing) + list(unexpected)
               if "position_ids" not in k and "token_type_ids" not in k]
        if bad:
            raise RuntimeError(f"веса не совпали с моделью: {bad}")
        self.model.eval()

    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        return predict_texts(self.model, self.encoder, self.labels, texts)
