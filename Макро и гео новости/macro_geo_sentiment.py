"""
macro_geo_sentiment.py

Добавляет GeRaCl-сентимент к macro/geo новостям из macro_geo_signals.csv.
2 GPU параллельно через multiprocessing.fork.

Вход:  macro_geo_signals.csv (из датасета Kaggle)
Выход: macro_geo_signals.csv (в /kaggle/working) + колонки:
         sentiment_neg, sentiment_neu, sentiment_pos, sentiment
"""

import subprocess
import sys
import os
import glob as _glob

import pandas as pd
import numpy as np
import multiprocessing as mp

# ─── Установка пакетов ───────────────────────────────────────────────────────

def install_package(name, pip_name=None):
    try:
        __import__(name)
    except ImportError:
        print(f"Устанавливаю {pip_name or name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or name, "-q"])

install_package("loguru")
install_package("pytorch_lightning", "pytorch-lightning")

try:
    import geracl
except ImportError:
    print("Устанавливаю geracl из GitHub...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/deepvk/GeRaCl.git", "-q"
    ])

# ─── Пути ────────────────────────────────────────────────────────────────────

_SIGNALS_CANDIDATES = [
    "/kaggle/input/datasets/markbabii/macro-geo-signals/macro_geo_signals.csv",
    "/kaggle/input/macro-geo-signals/macro_geo_signals.csv",
    "/kaggle/input/telegram-messages-db/macro_geo_signals.csv",
    "/kaggle/working/macro_geo_signals.csv",
    "macro_geo_signals.csv",
]
INPUT_CSV = next((p for p in _SIGNALS_CANDIDATES if os.path.exists(p)), "macro_geo_signals.csv")

WORK_DIR   = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
OUTPUT_CSV = os.path.join(WORK_DIR, "macro_geo_signals.csv")

print(f"Входной файл:  {INPUT_CSV}")
print(f"Выходной файл: {OUTPUT_CSV}")

# ─── Настройки ───────────────────────────────────────────────────────────────

MODEL_NAME       = "deepvk/GeRaCl-USER2-base"
LABEL_SPACE      = ["негативный", "нейтральный", "позитивный"]
BATCH_SIZE       = 64    # строк за одну итерацию GPU
CHECKPOINT_EVERY = 1000  # строк между сохранениями чекпоинта
MAX_TEXT_LEN     = 512   # символов текста

# ─── Worker ──────────────────────────────────────────────────────────────────

def worker_sentiment(gpu_id: int, df_rows: list, out_csv: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from transformers import AutoTokenizer
    from geracl import GeraclHF
    from geracl.data.batch_creation import prepare_inference_batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU {gpu_id}] Загрузка GeRaCl на {device}...")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = GeraclHF.from_pretrained(MODEL_NAME).to(device).eval()
    if hasattr(model, "_classification_core"):
        model._classification_core._device = device
    print(f"[GPU {gpu_id}] Модель загружена. Задач: {len(df_rows):,}")
    sys.stdout.flush()

    # Чекпоинт
    done_ids: set = set()
    if os.path.exists(out_csv):
        tmp = pd.read_csv(out_csv, usecols=["message_id", "channel"])
        done_ids = set(zip(tmp["message_id"], tmp["channel"]))
        print(f"[GPU {gpu_id}] Чекпоинт: {len(done_ids):,} уже обработано.")

    todo = [r for r in df_rows if (r["message_id"], r["channel"]) not in done_ids]
    print(f"[GPU {gpu_id}] Осталось: {len(todo):,}")
    sys.stdout.flush()

    if not todo:
        return

    buffer = []
    first_write = not os.path.exists(out_csv)
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, batch_start in enumerate(range(0, len(todo), BATCH_SIZE)):
        batch = todo[batch_start: batch_start + BATCH_SIZE]
        texts = [str(r["message"])[:MAX_TEXT_LEN] for r in batch]

        with torch.no_grad():
            input_ids, attention_mask, classes_mask, classes_count = prepare_inference_batch(
                texts, [LABEL_SPACE] * len(texts), tokenizer
            )
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            classes_mask   = classes_mask.to(device)
            classes_count  = classes_count.to(device)

            similarities = model(input_ids, attention_mask, classes_mask, classes_count)

            # Разбиваем similarities на probabilities для каждого текста
            batch_probs = []
            start_idx = 0
            for count in classes_count:
                end_idx = start_idx + count.item()
                text_sims  = similarities[start_idx:end_idx]
                text_probs = torch.softmax(text_sims, dim=0).cpu().tolist()
                batch_probs.append(text_probs)
                start_idx = end_idx

            # Очистка памяти
            del input_ids, attention_mask, classes_mask, classes_count, similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for row, probs in zip(batch, batch_probs):
            neg, neu, pos = probs[0], probs[1], probs[2]
            sentiment = 0.0 if neu > 0.5 else round(pos - neg, 4)
            buffer.append({
                "message_id":    row["message_id"],
                "channel":       row["channel"],
                "sentiment_neg": round(neg, 4),
                "sentiment_neu": round(neu, 4),
                "sentiment_pos": round(pos, 4),
                "sentiment":     sentiment,
            })

        if (i + 1) % 50 == 0 or i == n_batches - 1:
            done = min(batch_start + BATCH_SIZE, len(todo))
            pct  = 100 * done / len(todo)
            bar  = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
            print(f"[GPU {gpu_id}] |{bar}| {done:,}/{len(todo):,} ({pct:.1f}%)", flush=True)

        if len(buffer) >= CHECKPOINT_EVERY:
            pd.DataFrame(buffer).to_csv(out_csv, mode="a", header=first_write, index=False)
            first_write = False
            buffer.clear()

    if buffer:
        pd.DataFrame(buffer).to_csv(out_csv, mode="a", header=first_write, index=False)

    print(f"[GPU {gpu_id}] Готово.")
    sys.stdout.flush()


# ─── Параллельный запуск ─────────────────────────────────────────────────────

def _run_parallel(df_rows: list, out_csvs: list):
    n    = len(df_rows)
    half = n // 2
    splits = [df_rows[:half], df_rows[half:]]

    procs = []
    for gpu_id in range(2):
        p = mp.Process(target=worker_sentiment, args=(gpu_id, splits[gpu_id], out_csvs[gpu_id]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    for i, p in enumerate(procs):
        if p.exitcode != 0:
            raise RuntimeError(f"Worker GPU{i} завершился с ошибкой (exitcode={p.exitcode})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено: {len(df):,} строк")

    # Чекпоинт: уже обработанные записи
    done_ids: set = set()
    for g in (0, 1):
        path = os.path.join(WORK_DIR, f"sentiment_gpu{g}.csv")
        if os.path.exists(path):
            tmp = pd.read_csv(path, usecols=["message_id", "channel"])
            done_ids.update(zip(tmp["message_id"], tmp["channel"]))
    # Если OUTPUT_CSV уже содержит сентимент (перезапуск после мержа)
    if os.path.exists(OUTPUT_CSV):
        cols = pd.read_csv(OUTPUT_CSV, nrows=0).columns.tolist()
        if "sentiment" in cols:
            tmp = pd.read_csv(OUTPUT_CSV, usecols=["message_id", "channel"])
            done_ids.update(zip(tmp["message_id"], tmp["channel"]))

    df_todo = df[
        ~pd.Series(list(zip(df["message_id"], df["channel"])), index=df.index).isin(done_ids)
    ].copy().reset_index(drop=True)
    print(f"Уже обработано: {len(done_ids):,} | Осталось: {len(df_todo):,}")

    if not df_todo.empty:
        out_sent = [os.path.join(WORK_DIR, f"sentiment_gpu{g}.csv") for g in range(2)]
        _run_parallel(df_todo.to_dict("records"), out_sent)

    # ── Мерж sentiment_gpu*.csv ──
    sent_files = sorted(_glob.glob(os.path.join(WORK_DIR, "sentiment_gpu*.csv")))
    if not sent_files:
        print("Нет файлов sentiment_gpu*.csv — нечего мержить.")
        return

    sent_parts = [pd.read_csv(f) for f in sent_files]
    df_sent = pd.concat(sent_parts, ignore_index=True).drop_duplicates(subset=["message_id", "channel"])
    print(f"Сентимент собран: {len(df_sent):,} записей")

    # Джойним сентимент к основному файлу
    # Убираем старые колонки сентимента если есть
    for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df_merged = df.merge(
        df_sent[["message_id", "channel", "sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment"]],
        on=["message_id", "channel"],
        how="left"
    )
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nСохранено: {len(df_merged):,} строк → {OUTPUT_CSV}")
    print(f"Без сентимента: {df_merged['sentiment'].isna().sum()} строк")

    # ── Статистика ──
    print("\n=== Распределение сентимента ===")
    bins       = [-1.01, -0.1, 0.1, 1.01]
    bin_labels = ["негативный (<-0.1)", "нейтральный (-0.1..+0.1)", "позитивный (>+0.1)"]
    df_merged["_sent_label"] = pd.cut(
        df_merged["sentiment"].astype(float), bins=bins, labels=bin_labels
    )
    print(df_merged["_sent_label"].value_counts())

    print("\n=== Примеры (по 2 на каждый тип) ===")
    for label in bin_labels:
        subset = df_merged[df_merged["_sent_label"] == label].head(2)
        for _, r in subset.iterrows():
            print(f"[{r['category']}][{r['scope']}] sentiment={r['sentiment']:+.3f}  "
                  f"neg={r['sentiment_neg']:.3f} neu={r['sentiment_neu']:.3f} pos={r['sentiment_pos']:.3f}")
            print(f"  {str(r['message'])[:120].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
