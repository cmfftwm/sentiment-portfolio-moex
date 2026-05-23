import sqlite3
import pandas as pd
import os

DB_PATH  = "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages_new.db"
CSV_PATH = "/Users/markbabii/PycharmProjects/analiz_santiment/гео и макро 20.11.2025/Сентимент/macro_geo_signals.csv"

print("=" * 60)
print("Загрузка macro_geo_signals → news_market_sentiment + news_sector_sentiment")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    print(f"CSV не найден: {CSV_PATH}")
    exit(1)

if not os.path.exists(DB_PATH):
    print(f"БД не найдена: {DB_PATH}")
    exit(1)

# ─── Загрузка CSV ─────────────────────────────────────────────────────────────

print(f"\nЗагружаю {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"   Всего строк: {len(df):,}")
print(f"   Колонки: {df.columns.tolist()}")

df_market = df[df["scope"] == "market"].copy().sort_values("date").reset_index(drop=True)
df_sector = df[df["scope"] == "sector"].copy().sort_values("date").reset_index(drop=True)
print(f"   scope=market: {len(df_market):,}")
print(f"   scope=sector: {len(df_sector):,}")

# ─── Подключение к БД ─────────────────────────────────────────────────────────

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ─── news_market_sentiment ────────────────────────────────────────────────────

print("\nСоздаю таблицу news_market_sentiment...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_market_sentiment (
        channel       TEXT,
        message_id    INTEGER,
        date          TIMESTAMP,
        category      TEXT,
        sentiment     REAL,
        sentiment_neg REAL,
        sentiment_neu REAL,
        sentiment_pos REAL,
        PRIMARY KEY (channel, message_id)
    )
""")
conn.commit()

cursor.execute("SELECT COUNT(*) FROM news_market_sentiment")
old_count = cursor.fetchone()[0]
print(f"   Текущих записей: {old_count:,}")

cursor.execute("DELETE FROM news_market_sentiment")
conn.commit()

print(f"   Записываю {len(df_market):,} строк...")
BATCH = 1000
for i in range(0, len(df_market), BATCH):
    batch = df_market.iloc[i:i + BATCH]
    cursor.executemany("""
        INSERT INTO news_market_sentiment
            (channel, message_id, date, category,
             sentiment, sentiment_neg, sentiment_neu, sentiment_pos)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (r.channel, int(r.message_id), r.date, r.category,
         float(r.sentiment), float(r.sentiment_neg),
         float(r.sentiment_neu), float(r.sentiment_pos))
        for r in batch.itertuples()
    ])
    if (i + BATCH) % 5000 == 0 or i + BATCH >= len(df_market):
        conn.commit()
        print(f"   {min(i + BATCH, len(df_market)):,} / {len(df_market):,}")

cursor.execute("SELECT COUNT(*) FROM news_market_sentiment")
print(f"   Записано: {cursor.fetchone()[0]:,}")

# ─── news_sector_sentiment ────────────────────────────────────────────────────

print("\nСоздаю таблицу news_sector_sentiment...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_sector_sentiment (
        channel          TEXT,
        message_id       INTEGER,
        date             TIMESTAMP,
        category         TEXT,
        affected_sectors TEXT,
        affected_tickers TEXT,
        sentiment        REAL,
        sentiment_neg    REAL,
        sentiment_neu    REAL,
        sentiment_pos    REAL,
        PRIMARY KEY (channel, message_id)
    )
""")
conn.commit()

cursor.execute("SELECT COUNT(*) FROM news_sector_sentiment")
old_count = cursor.fetchone()[0]
print(f"   Текущих записей: {old_count:,}")

cursor.execute("DELETE FROM news_sector_sentiment")
conn.commit()

print(f"   Записываю {len(df_sector):,} строк...")
for i in range(0, len(df_sector), BATCH):
    batch = df_sector.iloc[i:i + BATCH]
    cursor.executemany("""
        INSERT INTO news_sector_sentiment
            (channel, message_id, date, category,
             affected_sectors, affected_tickers,
             sentiment, sentiment_neg, sentiment_neu, sentiment_pos)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (r.channel, int(r.message_id), r.date, r.category,
         str(r.affected_sectors) if pd.notna(r.affected_sectors) else None,
         str(r.affected_tickers) if pd.notna(r.affected_tickers) else None,
         float(r.sentiment), float(r.sentiment_neg),
         float(r.sentiment_neu), float(r.sentiment_pos))
        for r in batch.itertuples()
    ])
    if (i + BATCH) % 5000 == 0 or i + BATCH >= len(df_sector):
        conn.commit()
        print(f"   {min(i + BATCH, len(df_sector)):,} / {len(df_sector):,}")

cursor.execute("SELECT COUNT(*) FROM news_sector_sentiment")
print(f"   Записано: {cursor.fetchone()[0]:,}")

# ─── Итог ─────────────────────────────────────────────────────────────────────

conn.close()
print("\nГотово!")
print("   Таблицы в БД:")
print("   - news_market_sentiment  (scope=market)")
print("   - news_sector_sentiment  (scope=sector)")
print("   - news_ticker_sentiment  (корпоративные, уже были)")
