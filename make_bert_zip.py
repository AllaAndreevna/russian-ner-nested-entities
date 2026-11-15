"""Собирает submission_bert.zip: solution_bert.py кладётся в корень как solution.py,
код обучения - в папку training/ (там solution.py - это CRF-версия, как ждёт train.py).

    python make_bert_zip.py
"""
import os
import zipfile

OUT = "submission_bert.zip"

# (файл на диске, имя внутри архива)
FILES = [
    ("solution_bert.py", "solution.py"),
    ("bert_model.pt", "bert_model.pt"),
    ("bert_meta.json", "bert_meta.json"),
    ("bert_tokenizer.json", "bert_tokenizer.json"),
    ("docs/description_bert.txt", "description.txt"),
    ("docs/README_BERT.md", "README.md"),
    ("bert_train.py", "training/bert_train.py"),
    ("solution_bert.py", "training/solution_bert.py"),
    ("train.py", "training/train.py"),
    ("solution.py", "training/solution.py"),
    ("check_submission.py", "training/check_submission.py"),
    ("diagnose_vocab.py", "training/diagnose_vocab.py"),
]

missing = [src for src, _ in FILES if not os.path.exists(src)]
if missing:
    raise SystemExit(f"Нет файлов: {missing}")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in FILES:
        z.write(src, arc)

size = os.path.getsize(OUT) / 1024 / 1024
print(f"{OUT}: {size:.2f} МБ (лимит 16)")
if size > 16:
    print("ПРЕВЫШЕН ЛИМИТ: уменьшите --max-vocab / --short-len в bert_train.py и обучите заново")