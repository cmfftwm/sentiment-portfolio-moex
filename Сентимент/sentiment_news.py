"""
sentiment_news_kaggle.py

Добавляет GeRaCl-сентимент к новостям из news_with_tickers.db.
2 GPU параллельно через multiprocessing.fork.

Вход:  news_with_tickers.db (из датасета Kaggle)
Выход: news_sentiment.csv (в /kaggle/working) с колонками:
         row_id, source, section, ticker, news_id,
         publication_datetime, sentiment_neg, sentiment_neu,
         sentiment_pos, sentiment
"""

import subprocess
import sys
import os
import re
import glob as _glob
import sqlite3

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

WORK_DIR   = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
OUTPUT_CSV = os.path.join(WORK_DIR, "news_sentiment.csv")

# Ищем БД только в /kaggle/input/ (working — папка для вывода, не входных данных)
print("Ищу news_with_tickers.db...")
all_db_files = _glob.glob("/kaggle/input/**/*.db", recursive=True)
print(f"   Все .db файлы в /kaggle/input/: {all_db_files}")

_DB_CANDIDATES = [
    "/kaggle/input/news-with-tickers/news_with_tickers.db",
    "/kaggle/input/newstickers/news_with_tickers.db",
] + [f for f in all_db_files if "news_with_tickers" in f]

# Локальный fallback (для запуска вне Kaggle)
if not any(os.path.exists(p) for p in _DB_CANDIDATES):
    _DB_CANDIDATES += ["news_with_tickers.db"]

DB_PATH = next((p for p in _DB_CANDIDATES if os.path.exists(p)), None)

if DB_PATH is None:
    print("news_with_tickers.db не найдена ни в одном из мест.")
    print("   Убедитесь что датасет подключён в разделе 'Add Input' на Kaggle.")
    raise SystemExit(1)

# Проверяем что таблица news_ticker есть в БД
import sqlite3 as _sqlite3
_check = _sqlite3.connect(DB_PATH)
_tables = [r[0] for r in _check.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
_rows   = _check.execute("SELECT COUNT(*) FROM news_ticker").fetchone()[0] if "news_ticker" in _tables else 0
_check.close()
print(f"БД найдена:   {DB_PATH}")
print(f"  Таблицы:      {_tables}")
print(f"  Строк:        {_rows:,}")
if "news_ticker" not in _tables or _rows == 0:
    print("Таблица news_ticker пустая или отсутствует!")
    raise SystemExit(1)

print(f"Выходной файл: {OUTPUT_CSV}")


# ─── Настройки ───────────────────────────────────────────────────────────────

MODEL_NAME       = "deepvk/GeRaCl-USER2-base"
LABEL_SPACE      = ["негативный", "нейтральный", "позитивный"]
BATCH_SIZE       = 64
CHECKPOINT_EVERY = 1000
MAX_TEXT_LEN     = 512


# ─── Извлечение контекста вокруг тикера ──────────────────────────────────────

def extract_ticker_context(text: str, ticker: str, context_sentences: int = 1) -> str:
    if not text or not ticker:
        return text or ""

    parts = re.split(r"([.!?]+(?:\s+|$))", text)
    sentences = []
    for i in range(0, len(parts), 2):
        s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        s = s.strip()
        if s:
            sentences.append(s)

    if not sentences:
        return text

    pattern = rf"\b{re.escape(ticker)}\b"
    relevant = [i for i, s in enumerate(sentences) if re.search(pattern, s, re.IGNORECASE)]

    if not relevant:
        return text

    indices = set()
    for idx in relevant:
        indices.add(idx)
        for off in range(1, context_sentences + 1):
            if idx - off >= 0:
                indices.add(idx - off)
            if idx + off < len(sentences):
                indices.add(idx + off)

    context = " ".join(sentences[i] for i in sorted(indices)).strip()
    return context if len(context) >= 30 else text


# ─── Worker (запускается в отдельном процессе на каждом GPU) ─────────────────

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

    # Чекпоинт — пропускаем уже обработанные row_id
    done_ids: set = set()
    if os.path.exists(out_csv):
        tmp = pd.read_csv(out_csv, usecols=["row_id"])
        done_ids = set(tmp["row_id"].tolist())
        print(f"[GPU {gpu_id}] Чекпоинт: {len(done_ids):,} уже обработано.")

    todo = [r for r in df_rows if r["row_id"] not in done_ids]
    print(f"[GPU {gpu_id}] Осталось: {len(todo):,}")
    sys.stdout.flush()

    if not todo:
        return

    buffer = []
    first_write = not os.path.exists(out_csv)
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, batch_start in enumerate(range(0, len(todo), BATCH_SIZE)):
        batch = todo[batch_start: batch_start + BATCH_SIZE]

        # Формируем текст: title + контекст вокруг тикера
        texts = []
        for r in batch:
            full = f"{r.get('title', '') or ''} {r.get('text', '') or ''}".strip()
            context = extract_ticker_context(full, r["ticker"], context_sentences=1)
            texts.append(context[:MAX_TEXT_LEN])

        with torch.no_grad():
            input_ids, attention_mask, classes_mask, classes_count = prepare_inference_batch(
                texts, [LABEL_SPACE] * len(texts), tokenizer
            )
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            classes_mask   = classes_mask.to(device)
            classes_count  = classes_count.to(device)

            similarities = model(input_ids, attention_mask, classes_mask, classes_count)

            batch_probs = []
            start_idx = 0
            for count in classes_count:
                end_idx = start_idx + count.item()
                text_probs = torch.softmax(similarities[start_idx:end_idx], dim=0).cpu().tolist()
                batch_probs.append(text_probs)
                start_idx = end_idx

            del input_ids, attention_mask, classes_mask, classes_count, similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for row, probs in zip(batch, batch_probs):
            neg, neu, pos = probs[0], probs[1], probs[2]
            sentiment = 0.0 if neu > 0.5 else round(pos - neg, 4)
            buffer.append({
                "row_id":               row["row_id"],
                "source":               row["source"],
                "section":              row["section"],
                "ticker":               row["ticker"],
                "news_id":              row["news_id"],
                "publication_datetime": row["publication_datetime"],
                "sentiment_neg":        round(neg, 4),
                "sentiment_neu":        round(neu, 4),
                "sentiment_pos":        round(pos, 4),
                "sentiment":            sentiment,
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


# ─── Параллельный запуск на 2 GPU ────────────────────────────────────────────

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
    # Загружаем данные из SQLite
    print(f"\nЧитаю {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT id AS row_id, source, section, ticker,
               news_id, publication_datetime, title, text
        FROM news_ticker
        ORDER BY publication_datetime
    """, conn)
    conn.close()
    print(f"Загружено: {len(df):,} строк")

    # Чекпоинт — собираем уже обработанные row_id
    done_ids: set = set()
    for g in (0, 1):
        path = os.path.join(WORK_DIR, f"sentiment_gpu{g}.csv")
        if os.path.exists(path):
            tmp = pd.read_csv(path, usecols=["row_id"])
            done_ids.update(tmp["row_id"].tolist())
    if os.path.exists(OUTPUT_CSV):
        cols = pd.read_csv(OUTPUT_CSV, nrows=0).columns.tolist()
        if "sentiment" in cols:
            tmp = pd.read_csv(OUTPUT_CSV, usecols=["row_id"])
            done_ids.update(tmp["row_id"].tolist())

    df_todo = df[~df["row_id"].isin(done_ids)].copy().reset_index(drop=True)
    print(f"Уже обработано: {len(done_ids):,} | Осталось: {len(df_todo):,}")

    if not df_todo.empty:
        out_sent = [os.path.join(WORK_DIR, f"sentiment_gpu{g}.csv") for g in range(2)]
        _run_parallel(df_todo.to_dict("records"), out_sent)

    # ── Мерж sentiment_gpu*.csv ──────────────────────────────────────────────
    sent_files = sorted(_glob.glob(os.path.join(WORK_DIR, "sentiment_gpu*.csv")))
    if not sent_files:
        print("Нет файлов sentiment_gpu*.csv — нечего мержить.")
        return

    sent_parts = [pd.read_csv(f) for f in sent_files]
    df_sent = pd.concat(sent_parts, ignore_index=True).drop_duplicates(subset=["row_id"])
    print(f"Сентимент собран: {len(df_sent):,} записей")

    # Убираем старые колонки сентимента если есть
    for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df_merged = df.merge(
        df_sent[["row_id", "news_id", "sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment"]],
        on="row_id",
        how="left",
        suffixes=("", "_sent")
    )
    # Используем news_id из df_sent если в df он пустой
    if "news_id_sent" in df_merged.columns:
        df_merged["news_id"] = df_merged["news_id"].fillna(df_merged["news_id_sent"])
        df_merged.drop(columns=["news_id_sent"], inplace=True)
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nСохранено: {len(df_merged):,} строк → {OUTPUT_CSV}")
    print(f"Без сентимента: {df_merged['sentiment'].isna().sum()} строк")

    # ── Статистика ───────────────────────────────────────────────────────────
    print("\n=== Распределение сентимента ===")
    bins       = [-1.01, -0.1, 0.1, 1.01]
    bin_labels = ["негативный (<-0.1)", "нейтральный (-0.1..+0.1)", "позитивный (>+0.1)"]
    df_merged["_label"] = pd.cut(df_merged["sentiment"].astype(float), bins=bins, labels=bin_labels)
    print(df_merged["_label"].value_counts())

    print("\n=== По источникам ===")
    print(df_merged.groupby("source")["sentiment"].describe().round(3))


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
