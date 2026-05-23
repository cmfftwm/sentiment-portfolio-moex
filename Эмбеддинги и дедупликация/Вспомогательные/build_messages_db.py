"""
build_messages_db.py

Создаёт новую telegram_messages_new.db по аналогии с telegram_messages.db:
  - берёт уникальные (channel, message_id) из all_embeddings.parquet
  - берёт тексты из staging.db
  - создаёт таблицы messages_{channel} с колонками:
    message_number, message_id, date, message, tickers
"""

import sqlite3
import polars as pl
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR  = Path(__file__).parent
STAGING_DB  = SCRIPT_DIR / "staging.db"
EMBEDDINGS  = SCRIPT_DIR / "all_embeddings.parquet"
OUTPUT_DB   = SCRIPT_DIR / "telegram_messages_new.db"

print("=" * 60)
print("Создание telegram_messages_new.db")
print("=" * 60)

# Читаем уникальные message_id из all_embeddings.parquet
print("Читаю all_embeddings.parquet...")
emb_df = pl.read_parquet(EMBEDDINGS).select(["channel", "message_id"])
print(f"  Уникальных записей: {len(emb_df):,}")

# Нормализуем: убираем префикс messages_ для сравнения со staging.db
emb_df = emb_df.with_columns(
    pl.col("channel").str.replace("^messages_", "").alias("channel_plain")
)

# Множество (channel_plain, message_id) для фильтрации со staging.db
unique_pairs = set(zip(emb_df["channel_plain"].to_list(), emb_df["message_id"].to_list()))
# Каналы с префиксом messages_ для имён таблиц
channels = sorted(emb_df["channel"].unique().to_list())
print(f"  Каналов: {len(channels)}")

# Читаем staging.db
print("\nЧитаю staging.db...")
src_conn = sqlite3.connect(STAGING_DB)
rows = src_conn.execute(
    "SELECT channel, message_id, date, message FROM staging WHERE message IS NOT NULL"
).fetchall()
src_conn.close()
print(f"  Строк в staging: {len(rows):,}")

# Фильтруем только уникальные (те что прошли дедупликацию)
filtered = [(ch, mid, dt, msg) for ch, mid, dt, msg in rows if (ch, mid) in unique_pairs]
print(f"  После фильтрации: {len(filtered):,}")

# Группируем по каналу
from collections import defaultdict
by_channel = defaultdict(list)
for ch, mid, dt, msg in filtered:
    by_channel[ch].append((mid, dt, msg))

# Создаём выходную БД
if OUTPUT_DB.exists():
    OUTPUT_DB.unlink()

dst_conn = sqlite3.connect(OUTPUT_DB)
cursor = dst_conn.cursor()

total_inserted = 0
for channel in tqdm(channels, desc="Создаю таблицы"):
    table = channel  # уже содержит messages_ префикс
    channel_plain = channel.replace("messages_", "", 1)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table}" (
            message_number INTEGER,
            message_id     INTEGER,
            date           TIMESTAMP,
            message        TEXT,
            tickers        TEXT
        )
    """)

    rows_ch = sorted(by_channel[channel_plain], key=lambda x: x[1])  # сортировка по message_id
    data = [
        (i + 1, mid, dt, msg, None)
        for i, (mid, dt, msg) in enumerate(rows_ch)
    ]
    cursor.executemany(f'INSERT INTO "{table}" VALUES (?,?,?,?,?)', data)
    total_inserted += len(data)

dst_conn.commit()
dst_conn.close()

print(f"\n{'=' * 60}")
print(f"Готово!")
print(f"  Таблиц создано: {len(channels)}")
print(f"  Строк вставлено: {total_inserted:,}")
print(f"  Файл: {OUTPUT_DB}")
print(f"  Размер: {OUTPUT_DB.stat().st_size / 1024**2:.1f} MB")
print("=" * 60)
