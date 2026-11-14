"""Локальная проверка архива так, как это делает проверяющая система.

    python check_submission.py submission.zip data/dev
"""
import glob
import os
import sys
import tempfile
import time
import zipfile

zip_path = sys.argv[1] if len(sys.argv) > 1 else "submission.zip"
dev_dir = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "data/dev")  # до chdir!

print(f"размер архива: {os.path.getsize(zip_path) / 1024 / 1024:.2f} МБ (лимит 16)")
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    print("файлы в архиве:", names)
    if any("\\" in n for n in names):
        print("ОШИБКА: в путях архива обратные слэши (архив собран не тем инструментом)")
    if "solution.py" not in names:
        print("ОШИБКА: solution.py должен лежать в корне архива, а не во вложенной папке")
    need = ["solution.py", "description.txt"]
    if "bert_model.pt" in names:
        need += ["bert_model.pt", "bert_meta.json", "bert_tokenizer.json"]
    else:
        need += [f"model_{k}.crfsuite" for k in range(3)]
    missing = [n for n in need if n not in names]
    if missing:
        print("ОШИБКА: в архиве нет файлов:", missing)
    tmp = tempfile.mkdtemp()
    z.extractall(tmp)

sys.path.insert(0, tmp)
os.chdir(tmp)  # чтобы случайно не подхватить файлы из папки проекта
import solution  # noqa: E402

print("solution импортирован из:", solution.__file__)
texts = []
for p in sorted(glob.glob(os.path.join(dev_dir, "*.txt")))[:20]:
    with open(os.path.join(os.path.dirname(p), os.path.basename(p)), encoding="utf-8", newline="") as f:
        texts.append(f.read())

if not texts:
    raise SystemExit(f"Не найдены *.txt в {dev_dir}: проверьте путь к dev")

t0 = time.time()
sol = solution.Solution()
res = list(sol.predict(texts))
print(f"{len(texts)} текстов обработано за {time.time() - t0:.1f} с")

assert len(res) == len(texts), "число результатов не равно числу текстов"
for text, ents in zip(texts, res):
    assert isinstance(ents, set), "каждый результат должен быть set"
    for s, e, t in ents:
        assert isinstance(s, int) and isinstance(e, int) and isinstance(t, str)
        assert 0 <= s < e <= len(text)
print("формат ответа в порядке; пример:", sorted(res[0])[:3])
print("ПРОВЕРКА ПРОЙДЕНА")