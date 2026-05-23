"""
step2d_add_negpos.py

Добавляет в dataset_vol.parquet отдельные признаки негативного
и позитивного сентимента (вместо одного итогового sentiment):

  ticker_neg_7d  — rolling 7d mean sentiment_neg по тикеру
  ticker_pos_7d  — rolling 7d mean sentiment_pos по тикеру
  market_neg_7d  — rolling 7d mean sentiment_neg по рынку
  market_pos_7d  — rolling 7d mean sentiment_pos по рынку

Вход:  dataset_vol.parquet
Выход: dataset_vol_negpos.parquet
"""

import sqlite3
import os
import pandas as pd
import polars as pl

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

DB_PATH        = os.path.join(PROJECT_DIR, "telegram_messages.db")
INPUT_PARQUET  = os.path.join(BASE_DIR, "dataset_vol.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol_negpos.parquet")

print("=" * 60)
print("Шаг 2d: Добавление neg/pos сентиментных фичей")
print("=" * 60)

# ─── Загрузка базового датасета ───────────────────────────────────────────────

base = pl.read_parquet(INPUT_PARQUET)
print(f"Базовый датасет: {len(base):,} строк, {base['ticker'].n_unique()} тикеров")

# ─── Загрузка neg/pos из БД ───────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)

print("\nЗагружаю sentiment_neg/pos из news_ticker_sentiment...")
df_ticker = pl.read_database(
    """SELECT ticker, date, sentiment_neg, sentiment_pos
       FROM news_ticker_sentiment
       WHERE sentiment_neg IS NOT NULL AND sentiment_pos IS NOT NULL""",
    conn
)
print(f"  {len(df_ticker):,} строк")

print("Загружаю sentiment_neg/pos из news_market_sentiment...")
df_market = pl.read_database(
    """SELECT date, sentiment_neg, sentiment_pos
       FROM news_market_sentiment
       WHERE sentiment_neg IS NOT NULL AND sentiment_pos IS NOT NULL""",
    conn
)
print(f"  {len(df_market):,} строк")

conn.close()

# ─── Парсинг дат ─────────────────────────────────────────────────────────────

def parse_date_col(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    parsed = pl.Series(df[col].to_list()).str.to_datetime(strict=False, time_unit="us")
    if parsed.dtype.time_zone is not None:
        parsed = parsed.dt.replace_time_zone(None)
    return df.with_columns(parsed.alias(col)).with_columns(
        pl.col(col).dt.date().alias("date_only")
    )

df_ticker = parse_date_col(df_ticker)
df_market = parse_date_col(df_market)

# ─── 1. Дневной neg/pos по тикерам + rolling 7d ──────────────────────────────

print("\nАгрегирую neg/pos по тикерам...")
ticker_negpos = (
    df_ticker
    .group_by(["ticker", "date_only"])
    .agg([
        pl.mean("sentiment_neg").alias("ticker_neg_1d"),
        pl.mean("sentiment_pos").alias("ticker_pos_1d"),
    ])
    .sort(["ticker", "date_only"])
    .with_columns([
        pl.col("ticker_neg_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .over("ticker")
          .alias("ticker_neg_7d"),
        pl.col("ticker_pos_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .over("ticker")
          .alias("ticker_pos_7d"),
    ])
    .rename({"date_only": "date"})
    .select(["ticker", "date", "ticker_neg_7d", "ticker_pos_7d"])
)
print(f"  Тикеров: {ticker_negpos['ticker'].n_unique()}, строк: {len(ticker_negpos):,}")

# ─── 2. Дневной neg/pos по рынку + rolling 7d ────────────────────────────────

print("Агрегирую neg/pos по рынку...")
market_negpos = (
    df_market
    .group_by("date_only")
    .agg([
        pl.mean("sentiment_neg").alias("market_neg_1d"),
        pl.mean("sentiment_pos").alias("market_pos_1d"),
    ])
    .sort("date_only")
    .with_columns([
        pl.col("market_neg_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .alias("market_neg_7d"),
        pl.col("market_pos_1d")
          .rolling_mean(window_size=7, min_samples=1)
          .alias("market_pos_7d"),
    ])
    .rename({"date_only": "date"})
    .select(["date", "market_neg_7d", "market_pos_7d"])
)
print(f"  Дней: {len(market_negpos):,}")

# ─── 3. Джойн к базовому датасету ────────────────────────────────────────────

print("\nДжойню к базовому датасету...")

NEW_COLS = ["ticker_neg_7d", "ticker_pos_7d", "market_neg_7d", "market_pos_7d"]

dataset = base.join(ticker_negpos, on=["ticker", "date"], how="left")
dataset = dataset.join(market_negpos, on="date", how="left")

# Forward-fill до 7 дней — только по тикерным фичам, изолированно по ticker.
# market_neg_7d и market_pos_7d НЕ заполняем forward_fill: после
# sort(["ticker","date"]) глобальный fill протекает значения из 2025-го
# в 2019-й следующего по алфавиту тикера, создавая look-ahead bias.
# Для market_* достаточно fill_null(0.0) ниже — в 2019 их просто нет в данных.
dataset = dataset.sort(["ticker", "date"]).with_columns([
    pl.col(c).forward_fill(limit=7).over("ticker")
    for c in ["ticker_neg_7d", "ticker_pos_7d"]
])

# Заполняем нули там где нет данных
dataset = dataset.with_columns([
    pl.col(c).fill_null(0.0) for c in NEW_COLS
])

# ─── Статистика ──────────────────────────────────────────────────────────────

print(f"\nДатасет: {len(dataset):,} строк")
for col in NEW_COLS:
    nonzero = (dataset[col] != 0).mean()
    mean_val = dataset[col].mean()
    print(f"  {col:<20} покрытие={nonzero:.1%}, среднее={mean_val:.4f}")

# ─── Сохранение ──────────────────────────────────────────────────────────────

dataset.write_parquet(OUTPUT_PARQUET)
print(f"\nСохранено → {OUTPUT_PARQUET}")
print(f"Новые признаки: {NEW_COLS}")
