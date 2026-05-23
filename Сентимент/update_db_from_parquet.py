import sqlite3
import polars as pl
import os

DB_PATH = "telegram_messages_new.db"
PARQUET_PATH = "news_ticker_sentiment_new.parquet"

print("=" * 60)
print("Обновление БД из parquet файла")
print("=" * 60)

# Проверяем наличие файлов
if not os.path.exists(PARQUET_PATH):
    print(f"Parquet файл не найден: {PARQUET_PATH}")
    exit(1)

if not os.path.exists(DB_PATH):
    print(f"БД не найдена: {DB_PATH}")
    exit(1)

# Загружаем данные из parquet
print(f"\nЗагружаю данные из {PARQUET_PATH}...")
df = pl.read_parquet(PARQUET_PATH)
print(f"   Записей в parquet: {len(df):,}")

# Проверяем колонки
required_columns = ["channel", "message_id", "ticker", "date", "sentiment", "sentiment_neg", "sentiment_neu", "sentiment_pos"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Отсутствуют колонки: {missing_columns}")
    exit(1)

print(f"   Колонки: {df.columns}")

# Подключаемся к БД
print(f"\nПодключаюсь к БД: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Проверяем существование таблицы
cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name='news_ticker_sentiment'
""")
if not cursor.fetchone():
    print("Таблица news_ticker_sentiment не найдена в БД!")
    print("   Создаю таблицу...")
    cursor.execute("""
        CREATE TABLE news_ticker_sentiment (
            channel TEXT,
            message_id INTEGER,
            ticker TEXT,
            date TIMESTAMP,
            sentiment REAL,
            sentiment_neg REAL,
            sentiment_neu REAL,
            sentiment_pos REAL,
            PRIMARY KEY (channel, message_id, ticker)
        )
    """)
    conn.commit()
    print("   Таблица создана")

# Проверяем текущее количество записей
cursor.execute("SELECT COUNT(*) FROM news_ticker_sentiment")
old_count = cursor.fetchone()[0]
print(f"\nТекущее количество записей в БД: {old_count:,}")

# Очищаем таблицу
print(f"\nОчищаю таблицу news_ticker_sentiment...")
cursor.execute("DELETE FROM news_ticker_sentiment")
conn.commit()
print(f"   Таблица очищена")

# Записываем данные из parquet в БД
print(f"\nЗаписываю данные в БД...")
batch_size = 1000
total_rows = len(df)

for i in range(0, total_rows, batch_size):
    batch = df.slice(i, min(batch_size, total_rows - i))
    
    # Подготавливаем данные для вставки
    data_to_insert = []
    for row in batch.iter_rows(named=True):
        data_to_insert.append((
            row["channel"],
            int(row["message_id"]),
            row["ticker"],
            row["date"],
            float(row["sentiment"]),
            float(row["sentiment_neg"]),
            float(row["sentiment_neu"]),
            float(row["sentiment_pos"])
        ))
    
    # Вставляем батч
    cursor.executemany("""
        INSERT INTO news_ticker_sentiment 
        (channel, message_id, ticker, date, sentiment, sentiment_neg, sentiment_neu, sentiment_pos)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    
    if (i + batch_size) % 5000 == 0 or i + batch_size >= total_rows:
        conn.commit()
        print(f"   Обработано: {min(i + batch_size, total_rows):,} / {total_rows:,}")

# Финальный коммит
conn.commit()

# Проверяем результат
cursor.execute("SELECT COUNT(*) FROM news_ticker_sentiment")
new_count = cursor.fetchone()[0]
print(f"\nГотово!")
print(f"   Записано записей: {new_count:,}")
print(f"   Было записей: {old_count:,}")

conn.close()

print("\nОбновление завершено!")
