"""
Показывает дубликаты одной новости из РАЗНЫХ каналов
"""
import sys
import sqlite3
from pathlib import Path

import polars as pl

# Пути
SCRIPT_DIR = Path(__file__).parent.absolute()
UNIQUE_FILE = SCRIPT_DIR / "all_embeddings.parquet"
DUPS_FILE = SCRIPT_DIR / "all_embeddings_duplicates.parquet"
DB_PATH = SCRIPT_DIR / "telegram_messages.db"

# Проверки путей
for p, name in [(UNIQUE_FILE, "all_embeddings.parquet"), (DUPS_FILE, "all_embeddings_duplicates.parquet"), (DB_PATH, "telegram_messages.db")]:
    if not p.exists():
        print(f"Файл {p} не найден ({name})")
        sys.exit(1)

print("=" * 80)
print("Дубликаты из РАЗНЫХ каналов")
print("=" * 80)
print(f"Дубликаты:  {DUPS_FILE}")
print(f"БД:          {DB_PATH}\n")

# Читаем parquet
unique_df = pl.read_parquet(UNIQUE_FILE)
dups_df = pl.read_parquet(DUPS_FILE)

# Создаем словарь для быстрого поиска уникальных записей (для получения даты оригинала)
unique_dict = {}
for row in unique_df.iter_rows(named=True):
    key = (row["channel"], row["message_id"])
    unique_dict[key] = {
        "channel": row["channel"],
        "message_id": row["message_id"],
        "date": row["date"]
    }

# Фильтруем только дубликаты из разных каналов
cross_channel_dups = dups_df.filter(pl.col("channel") != pl.col("base_channel"))

print(f"Всего дубликатов: {len(dups_df):,}")
print(f"Дубликаты из разных каналов: {len(cross_channel_dups):,}\n")

if len(cross_channel_dups) == 0:
    print("Не найдено дубликатов из разных каналов!")
    print("   Все дубликаты находятся в одном канале.")
    print("\nВозможные причины:")
    print("   1. Данные отсортированы по каналам, а не по дате")
    print("   2. Новости действительно не дублируются между каналами")
    print("   3. Порог косинусного сходства слишком высокий")
    sys.exit(0)

# Подключаемся к БД
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def fetch_messages(channel, ids):
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    try:
        q = f"SELECT message_id, message FROM {channel} WHERE message_id IN ({placeholders})"
        cursor.execute(q, ids)
        return {mid: msg for mid, msg in cursor.fetchall()}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {}
        raise

# Показываем первые 20 примеров межканальных дубликатов
print("Показываю первые 20 примеров:\n")

for i, row in enumerate(cross_channel_dups.head(20).iter_rows(named=True), 1):
    orig_channel = row["base_channel"]
    dup_channel = row["channel"]
    orig_id = int(row["base_message_id"])
    dup_id = int(row["message_id"])
    
    # Получаем дату оригинала из unique_dict
    base_key = (orig_channel, orig_id)
    if base_key in unique_dict:
        orig_date = unique_dict[base_key]["date"]
    else:
        orig_date = "N/A"
    
    dup_date = row["date"]
    sim = row["sim"]
    dt_hours = row.get("dt", 0)

    # Тексты
    orig_text = fetch_messages(orig_channel, [orig_id]).get(orig_id, "<нет текста>")
    dup_text = fetch_messages(dup_channel, [dup_id]).get(dup_id, "<нет текста>")

    print("-" * 80)
    print(f"[{i}/20] Пример #{i}")
    print(f"Оригинал: {orig_channel} #{orig_id} ({orig_date})")
    print(f"Текст: {orig_text[:200]}..." if len(orig_text) > 200 else f"Текст: {orig_text}")
    print()
    print(f"Дубликат: {dup_channel} #{dup_id} ({dup_date})")
    print(f"Текст: {dup_text[:200]}..." if len(dup_text) > 200 else f"Текст: {dup_text}")
    print(f"cosine_sim: {sim:.4f} | Разница во времени: {dt_hours:.2f} часов")

conn.close()

print("\n" + "=" * 80)
print("Готово")
print("=" * 80)
