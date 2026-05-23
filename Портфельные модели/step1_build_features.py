"""
step1_build_features.py

Шаг 1: Агрегация сентимента в дневные фичи для каждого тикера.

Источники:
  - telegram_messages.db → news_ticker_sentiment, news_market_sentiment, news_sector_sentiment
  - tickers.json         → маппинг тикер → сектор

Выход:
  - Портфельная модель/features_daily.parquet
    Колонки:
      date, ticker, sector,
      ticker_sent_1d, ticker_sent_7d, ticker_sent_14d, ticker_sent_mom,
      ticker_news_count_1d, ticker_news_count_7d,
      market_sent_1d, market_sent_7d,
      sector_sent_1d, sector_sent_7d
"""

import sqlite3
import json
import os
import polars as pl

# ─── Пути ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

DB_PATH          = os.path.join(PROJECT_DIR, "telegram_messages.db")
TICKERS_JSON     = os.path.join(PROJECT_DIR, "tickers.json")
OUTPUT_PARQUET   = os.path.join(BASE_DIR, "features_daily.parquet")

print("=" * 60)
print("Шаг 1: Построение дневных фичей")
print("=" * 60)
print(f"БД:     {DB_PATH}")
print(f"Выход:  {OUTPUT_PARQUET}")

# ─── Загрузка tickers.json ───────────────────────────────────────────────────

with open(TICKERS_JSON, encoding="utf-8") as f:
    tickers_data = json.load(f)

ticker_sector = {
    (item.get("tiker") or item.get("ticker")): item.get("sector", "unknown")
    for item in tickers_data
}
print(f"\nТикеров в tickers.json: {len(ticker_sector)}")

# ─── Загрузка из БД ──────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)

print("\nЗагружаю news_ticker_sentiment...")
df_ticker = pl.read_database(
    "SELECT channel, message_id, ticker, date, sentiment FROM news_ticker_sentiment WHERE sentiment IS NOT NULL",
    conn
)
print(f"  {len(df_ticker):,} строк")

print("Загружаю news_market_sentiment...")
df_market = pl.read_database(
    "SELECT date, sentiment FROM news_market_sentiment WHERE sentiment IS NOT NULL",
    conn
)
print(f"  {len(df_market):,} строк")

print("Загружаю news_sector_sentiment...")
df_sector_raw = pl.read_database(
    "SELECT date, affected_sectors, sentiment FROM news_sector_sentiment WHERE sentiment IS NOT NULL",
    conn
)
print(f"  {len(df_sector_raw):,} строк")

conn.close()

# ─── Парсинг дат ─────────────────────────────────────────────────────────────

def parse_date_col(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    parsed = pl.Series(df[col].to_list()).str.to_datetime(strict=False, time_unit="us")
    if parsed.dtype.time_zone is not None:
        parsed = parsed.dt.replace_time_zone(None)
    return df.with_columns(parsed.alias(col)).with_columns(
        pl.col(col).dt.date().alias("date_only")
    )

df_ticker    = parse_date_col(df_ticker)
df_market    = parse_date_col(df_market)
df_sector_raw = parse_date_col(df_sector_raw)

# ─── 1. Дневной сентимент тикеров ────────────────────────────────────────────

print("\nАгрегирую тикерный сентимент по дням...")
ticker_daily = (
    df_ticker
    .group_by(["ticker", "date_only"])
    .agg([
        pl.mean("sentiment").alias("ticker_sent_1d"),
        pl.len().alias("ticker_news_count_1d"),
    ])
    .sort(["ticker", "date_only"])
)

# Rolling 7d и 14d — через join по дате диапазону
# Считаем через sort + rolling на уровне тикера
ticker_daily = ticker_daily.sort(["ticker", "date_only"])

# Добавляем rolling средние
ticker_daily = (
    ticker_daily
    .with_columns([
        pl.col("ticker_sent_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .over("ticker")
          .alias("ticker_sent_7d"),
        pl.col("ticker_sent_1d")
          .rolling_mean(window_size=14, min_samples=1)
          .over("ticker")
          .alias("ticker_sent_14d"),
        pl.col("ticker_news_count_1d")
          .rolling_sum(window_size=7, min_samples=1)
          .over("ticker")
          .alias("ticker_news_count_7d"),
    ])
    .with_columns(
        (pl.col("ticker_sent_1d") - pl.col("ticker_sent_7d")).alias("ticker_sent_mom")
    )
)

print(f"  Тикеров: {ticker_daily['ticker'].n_unique()}")
print(f"  Дней:    {ticker_daily['date_only'].n_unique()}")

# ─── 2. Дневной рыночный сентимент ───────────────────────────────────────────

print("\nАгрегирую рыночный сентимент по дням...")
market_daily = (
    df_market
    .group_by("date_only")
    .agg(pl.mean("sentiment").alias("market_sent_1d"))
    .sort("date_only")
    .with_columns(
        pl.col("market_sent_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .alias("market_sent_7d")
    )
)
print(f"  Дней: {len(market_daily)}")

# ─── 3. Дневной секторный сентимент ──────────────────────────────────────────

print("\nАгрегирую секторный сентимент по дням...")

# affected_sectors хранится как строка вида "['нефть и газ', 'банки']"
# Разворачиваем в отдельные строки
rows = []
for row in df_sector_raw.iter_rows(named=True):
    raw = row["affected_sectors"]
    if not raw:
        continue
    # Формат: "Нефть и газ" или "Ритейл, Химическая промышленность"
    sectors = [s.strip() for s in str(raw).split(",") if s.strip()]
    for s in sectors:
        rows.append({"date_only": row["date_only"], "sector": s, "sentiment": row["sentiment"]})

df_sector_expanded = pl.DataFrame(rows)
print(f"  Строк после разворачивания: {len(df_sector_expanded):,}")

sector_daily = (
    df_sector_expanded
    .group_by(["sector", "date_only"])
    .agg(pl.mean("sentiment").alias("sector_sent_1d"))
    .sort(["sector", "date_only"])
    .with_columns(
        pl.col("sector_sent_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .over("sector")
          .alias("sector_sent_7d")
    )
)
print(f"  Секторов: {sector_daily['sector'].n_unique()}")

# ─── 4. Добавляем сектор к тикерам ───────────────────────────────────────────

sector_map = pl.DataFrame([
    {"ticker": t, "sector": s} for t, s in ticker_sector.items()
])

ticker_daily = ticker_daily.join(sector_map, on="ticker", how="left").with_columns(
    pl.col("sector").fill_null("unknown")
)

# ─── 5. Джойним рыночный сентимент ───────────────────────────────────────────

ticker_daily = ticker_daily.join(market_daily, on="date_only", how="left")

# ─── 6. Джойним секторный сентимент ──────────────────────────────────────────

ticker_daily = ticker_daily.join(
    sector_daily,
    on=["sector", "date_only"],
    how="left"
)

# ─── 7. Финальный датасет ────────────────────────────────────────────────────

features = ticker_daily.rename({"date_only": "date"}).select([
    "date", "ticker", "sector",
    "ticker_sent_1d", "ticker_sent_7d", "ticker_sent_14d", "ticker_sent_mom",
    "ticker_news_count_1d", "ticker_news_count_7d",
    "market_sent_1d", "market_sent_7d",
    "sector_sent_1d", "sector_sent_7d",
])

features.write_parquet(OUTPUT_PARQUET)

print(f"\n{'=' * 60}")
print(f"Сохранено: {len(features):,} строк → {OUTPUT_PARQUET}")
print(f"Тикеров:   {features['ticker'].n_unique()}")
print(f"Период:    {features['date'].min()} → {features['date'].max()}")
print(f"Колонки:   {features.columns}")
print(f"\nПример:")
print(features.head(5))
