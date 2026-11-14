# NER на NEREL v1.1: rubert-tiny2 со слоями вложенности

## Воспроизведение
1. Python 3.10, зависимости:
   `pip install torch transformers tokenizers huggingface_hub razdel nltk sklearn-crfsuite`
   (путь к проекту только латиницей).
2. Перейти в папку `training/`. Положить NEREL v1.1 (github.com/nerel-ds/NEREL): папки `train` и `dev`
   с файлами `*.txt` и `*.ann` в `training/data/train` и `training/data/dev`. Папку `test` не использовать.
3. Нужен интернет один раз: скрипт скачивает публичную модель `cointegrated/rubert-tiny2` с Hugging Face.
4. Подбор числа эпох на dev: `python bert_train.py --data-dir data`
   (по умолчанию 10 эпох, `--max-vocab 10000`, `--short-len 2`); скрипт печатает лучшую эпоху N.
5. Финальные веса на train+dev: `python bert_train.py --data-dir data --train-on-dev --epochs N`
6. Результат: `bert_model.pt`, `bert_meta.json`, `bert_tokenizer.json` рядом со скриптом.
   Для проверяющей системы они кладутся в корень архива вместе с `solution.py`
   (это `solution_bert.py`; см. `make_bert_zip.py`).
7. (Необязательно) CRF-версия для сравнения: `python train.py --data-dir data`.
