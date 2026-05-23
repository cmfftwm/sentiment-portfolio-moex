"""
Вывод пар: уникальное сообщение и его дубликаты.
Источники:
- all_embeddings.parquet — уникальные записи (после дедупликации)
- all_embeddings_duplicates.parquet — дубликаты с ссылкой на оригинал
- telegram_messages.db — тексты сообщений

Как использовать:
python show_duplicate_pairs.py

Результат: печатает ВСЕ пары с полями канал/ID/дата/текст и косинусное сходство.
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

# Показываем все пары (без ограничений)
MAX_PAIRS = 10  # None = показать все пары

# Проверки путей
for p, name in [(UNIQUE_FILE, "all_embeddings.parquet"), (DUPS_FILE, "all_embeddings_duplicates.parquet"), (DB_PATH, "telegram_messages.db")]:
    if not p.exists():
        print(f"Файл {p} не найден ({name})")
        sys.exit(1)

print("=" * 80)
print("Пары оригинал ↔ дубликат")
print("=" * 80)
print(f"Уникальные: {UNIQUE_FILE}")
print(f"Дубликаты:  {DUPS_FILE}")
print(f"БД:          {DB_PATH}\n")

# Читаем parquet
unique_df = pl.read_parquet(UNIQUE_FILE)
dups_df = pl.read_parquet(DUPS_FILE)

# Создаем словарь для быстрого поиска уникальных записей
unique_dict = {}
for row in unique_df.iter_rows(named=True):
    key = (row["channel"], row["message_id"])
    unique_dict[key] = {
        "channel": row["channel"],
        "message_id": row["message_id"],
        "date": row["date"]
    }

# Подключаемся к БД, чтобы вытащить тексты
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

# Собираем ВСЕ пары (оригинал ↔ дубликат)
pairs = []
for row in dups_df.iter_rows(named=True):
    base_key = (row["base_channel"], row["base_message_id"])
    if base_key in unique_dict:
        orig_info = unique_dict[base_key]
        pairs.append({
            "orig_channel": orig_info["channel"],
            "orig_id": orig_info["message_id"],
            "orig_date": orig_info["date"],
            "dup_channel": row["channel"],
            "dup_id": row["message_id"],
            "dup_date": row["date"],
            "sim": row["sim"]
        })

print(f"Всего найдено пар: {len(pairs):,}")
print(f"Показываю все пары (orig ↔ dup):\n")

for i, pair in enumerate(pairs, 1):
    orig_channel = pair["orig_channel"]
    dup_channel = pair["dup_channel"]
    orig_id = int(pair["orig_id"])
    dup_id = int(pair["dup_id"])
    orig_date = pair["orig_date"]
    dup_date = pair["dup_date"]
    sim = pair["sim"]

    # Тексты
    orig_text = fetch_messages(orig_channel, [orig_id]).get(orig_id, "<нет текста>")
    dup_text = fetch_messages(dup_channel, [dup_id]).get(dup_id, "<нет текста>")

    print("-" * 80)
    print(f"[{i}/{len(pairs)}] Пара #{i}")
    print(f"Оригинал: {orig_channel} #{orig_id} ({orig_date})")
    print(f"Текст: {orig_text}")
    print()
    print(f"Дубликат: {dup_channel} #{dup_id} ({dup_date})")
    print(f"Текст: {dup_text}")
    print(f"cosine_sim: {sim:.3f}")

conn.close()

print("\nГотово")
