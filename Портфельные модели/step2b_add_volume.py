"""
step2b_add_volume.py

Обогащаем dataset.parquet объёмными фичами из moex_1m.db:
  volume_1d       — дневной оборот в рублях
  volume_ratio_5d — volume_1d / rolling_mean(5d)  (аномальный объём краткосрочный)
  volume_ratio_20d— volume_1d / rolling_mean(20d) (аномальный объём среднесрочный)

Вход:  dataset.parquet
Выход: dataset_vol.parquet   (НЕ перезаписывает оригинал)
"""

import sqlite3
import os
import polars as pl
from tqdm import tqdm

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

PRICES_DB      = os.path.join(PROJECT_DIR, "moex_1m.db")
INPUT_PARQUET  = os.path.join(BASE_DIR, "dataset.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol.parquet")

print("=" * 60)
print("Шаг 2b: Добавление объёмных фичей")
print("=" * 60)

dataset = pl.read_parquet(INPUT_PARQUET)
tickers = dataset["ticker"].unique().to_list()
print(f"Тикеров: {len(tickers)}, строк: {len(dataset):,}")

# ─── Загрузка дневного объёма из moex_1m.db ──────────────────────────────────

print("\nЗагружаю дневные объёмы из moex_1m.db...")
conn = sqlite3.connect(PRICES_DB)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
available = {r[0].lower(): r[0] for r in cursor.fetchall()}

vol_list = []
for ticker in tqdm(tickers, desc="Объёмы"):
    table = available.get(ticker.lower())
    if not table:
        continue
    df2 = pl.read_database(
        f'SELECT date(time) AS date, SUM(value) AS volume_1d '
        f'FROM "{table}" WHERE value IS NOT NULL '
        f'GROUP BY date(time) ORDER BY date(time)',
        conn
    )
    df2 = df2.with_columns([
        pl.col("date").cast(pl.Date),
        pl.lit(ticker).alias("ticker"),
    ])
    vol_list.append(df2)

conn.close()

vol = pl.concat(vol_list).sort(["ticker", "date"])
print(f"Объёмов: {len(vol):,} строк")

# ─── Объёмные фичи ───────────────────────────────────────────────────────────

vol = vol.sort(["ticker", "date"]).with_columns([
    pl.col("volume_1d")
      .rolling_mean(window_size=5,  min_samples=3)
      .over("ticker").alias("_vol_ma5"),
    pl.col("volume_1d")
      .rolling_mean(window_size=20, min_samples=10)
      .over("ticker").alias("_vol_ma20"),
]).with_columns([
    (pl.col("volume_1d") / (pl.col("_vol_ma5")  + 1e-9)).alias("volume_ratio_5d"),
    (pl.col("volume_1d") / (pl.col("_vol_ma20") + 1e-9)).alias("volume_ratio_20d"),
]).drop(["_vol_ma5", "_vol_ma20"])

# ─── Джойн к датасету ────────────────────────────────────────────────────────

dataset = dataset.join(vol, on=["ticker", "date"], how="left")

# Заполняем пропуски нулями (тикеры без объёма в БД)
dataset = dataset.with_columns([
    pl.col("volume_1d").fill_null(0.0),
    pl.col("volume_ratio_5d").fill_null(1.0),
    pl.col("volume_ratio_20d").fill_null(1.0),
])

print(f"\nДатасет с объёмами: {len(dataset):,} строк")
print(f"Новые колонки: volume_1d, volume_ratio_5d, volume_ratio_20d")
coverage = (dataset["volume_1d"] > 0).mean()
print(f"Покрытие объёмом: {coverage:.1%} строк")

dataset.write_parquet(OUTPUT_PARQUET)
print(f"\nСохранено → {OUTPUT_PARQUET}")
