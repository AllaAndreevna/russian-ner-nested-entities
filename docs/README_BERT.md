# NER на NEREL v1.1: rubert-tiny2 со слоями вложенности

## Воспроизведение
1. Python 3.10, зависимости:
   `pip install torch transformers tokenizers huggingface_hub razdel nltk sklearn-crfsuite`
   (путь к проекту только латиницей).
2. Перейти в папку `training/`. Положить NEREL v1.1 (github.com/nerel-ds/NEREL): папки `train` и `dev`
   с файлами `*.txt` и `*.ann` в `training/data/train` и `training/data/dev`. Папку `test` не использовать.
3. Нужен интернет один раз: скрипт скачивает публичную модель `cointegrated/rubert-tiny2` с Hugging Face.
4. Подбор числа эпох на dev (результаты не сохраняются), например:
   `python bert_train.py --data-dir data --epochs 15 --lr 3e-4 --batch-size 16 --ru-en-vocab --case-emb --emb-rank R --quantize-emb --no-save`
   Скрипт печатает лучшую эпоху N.
5. Финальные веса на train+dev с теми же флагами:
   `python bert_train.py --data-dir data --epochs N --lr 3e-4 --batch-size 16 --ru-en-vocab --case-emb --emb-rank R --quantize-emb --train-on-dev`
   Точные параметры итогового запуска записаны в `bert_meta.json` (ключ `train_args`).
6. Результат: `bert_model.pt`, `bert_meta.json`, `bert_tokenizer.json` рядом со скриптом.
   Для проверяющей системы они кладутся в корень архива вместе с `solution.py`
   (это `solution_bert.py`; см. `make_bert_zip.py`).
7. (Необязательно) CRF-версия для сравнения: `python train.py --data-dir data`.