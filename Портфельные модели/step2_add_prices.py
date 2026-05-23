"""
step2_add_prices.py

Шаг 2: Добавляем дневные цены, расширяем на все торговые дни,
        forward-fill сентимент и добавляем ценовые фичи.

Вход:  features_daily.parquet
Выход: dataset.parquet
  Колонки: все фичи + close, ret_1d, ret_5d, ret_10d, vol_10d, target
"""

import sqlite3
import os
import polars as pl
from tqdm import tqdm

# ─── Пути ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

PRICES_DB      = os.path.join(PROJECT_DIR, "moex_1m.db")
INPUT_PARQUET  = os.path.join(BASE_DIR, "features_daily.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "dataset.parquet")

print("=" * 60)
print("Шаг 2: Добавление цен и целевой переменной")
print("=" * 60)

# ─── Загрузка фичей ──────────────────────────────────────────────────────────

features = pl.read_parquet(INPUT_PARQUET)
tickers  = features["ticker"].unique().to_list()
print(f"Тикеров: {len(tickers)}, строк фичей: {len(features):,}")

SENT_COLS = [
    "ticker_sent_1d", "ticker_sent_7d", "ticker_sent_14d", "ticker_sent_mom",
    "ticker_news_count_1d", "ticker_news_count_7d",
    "market_sent_1d", "market_sent_7d",
    "sector_sent_1d", "sector_sent_7d",
]

# ─── Загрузка дневных цен из moex_1m.db ──────────────────────────────────────

print("\nЗагружаю дневные цены из moex_1m.db...")
conn = sqlite3.connect(PRICES_DB)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
available = {r[0].lower(): r[0] for r in cursor.fetchall()}

prices_list = []
for ticker in tqdm(tickers, desc="Цены"):
    table = available.get(ticker.lower())
    if not table:
        continue
    df2 = pl.read_database(
        f"SELECT date(time) AS date, close FROM \"{table}\" WHERE close IS NOT NULL ORDER BY time",
        conn
    )
    daily = (
        df2
        .with_columns(pl.col("date").cast(pl.Date))
        .group_by("date")
        .agg(pl.last("close").alias("close"))
        .with_columns(pl.lit(ticker).alias("ticker"))
    )
    prices_list.append(daily)

conn.close()

prices = pl.concat(prices_list).sort(["ticker", "date"])
print(f"Цен: {len(prices):,} строк, {prices['ticker'].n_unique()} тикеров")

# ─── Ценовые фичи ────────────────────────────────────────────────────────────

print("\nСчитаю ценовые фичи...")
prices = prices.sort(["ticker", "date"]).with_columns([
    # Доходности
    ((pl.col("close") - pl.col("close").shift(1).over("ticker")) /
      pl.col("close").shift(1).over("ticker")).alias("ret_1d"),
    ((pl.col("close") - pl.col("close").shift(5).over("ticker")) /
      pl.col("close").shift(5).over("ticker")).alias("ret_5d"),
    ((pl.col("close") - pl.col("close").shift(10).over("ticker")) /
      pl.col("close").shift(10).over("ticker")).alias("ret_10d"),
    ((pl.col("close") - pl.col("close").shift(20).over("ticker")) /
      pl.col("close").shift(20).over("ticker")).alias("ret_20d"),
    ((pl.col("close") - pl.col("close").shift(63).over("ticker")) /
      pl.col("close").shift(63).over("ticker")).alias("ret_63d"),
    # Скользящие средние
    pl.col("close").rolling_mean(window_size=5,  min_samples=3).over("ticker").alias("_ma5"),
    pl.col("close").rolling_mean(window_size=20, min_samples=10).over("ticker").alias("_ma20"),
    pl.col("close").rolling_mean(window_size=60, min_samples=30).over("ticker").alias("_ma60"),
    # RSI компоненты (промежуточные)
    pl.when(pl.col("close").shift(1).over("ticker").is_not_null())
      .then(pl.col("close") - pl.col("close").shift(1).over("ticker"))
      .otherwise(pl.lit(0.0)).alias("_delta"),
]).with_columns([
    # Волатильность
    pl.col("ret_1d").rolling_std(window_size=10, min_samples=5).over("ticker").alias("vol_10d"),
    pl.col("ret_1d").rolling_std(window_size=20, min_samples=10).over("ticker").alias("vol_20d"),
    # Std20 для Боллинджера
    pl.col("close").rolling_std(window_size=20, min_samples=10).over("ticker").alias("_std20"),
    # Целевая переменная: доходность следующего дня (1d)
    ((pl.col("close").shift(-1).over("ticker") - pl.col("close")) /
      pl.col("close")).alias("target"),
    # Целевая переменная 5d (для LightGBM с меньшим шумом)
    ((pl.col("close").shift(-5).over("ticker") - pl.col("close")) /
      pl.col("close")).alias("target_5d"),
    # RSI компоненты
    pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(pl.lit(0.0)).alias("_gain"),
    pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(pl.lit(0.0)).alias("_loss"),
]).with_columns([
    # Усреднённые gain/loss для RSI
    pl.col("_gain").rolling_mean(window_size=14, min_samples=7).over("ticker").alias("_avg_gain"),
    pl.col("_loss").rolling_mean(window_size=14, min_samples=7).over("ticker").alias("_avg_loss"),
]).with_columns([
    # RSI_14
    (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-9))).alias("rsi_14"),
    # MA ratios (тренд)
    (pl.col("_ma5")  / (pl.col("_ma20") + 1e-9)).alias("ma_ratio_5_20"),
    (pl.col("_ma20") / (pl.col("_ma60") + 1e-9)).alias("ma_ratio_20_60"),
    # Bollinger band position: (close - MA20) / (2 * std20)
    ((pl.col("close") - pl.col("_ma20")) / (2 * pl.col("_std20") + 1e-9)).alias("bb_pos"),
]).drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss", "_ma5", "_ma20", "_ma60", "_std20"])

# ─── Джойн сентимента ко всем торговым дням (asof) ───────────────────────────

print("\nДелаю forward-fill сентимента на все торговые дни...")

sent_cols = features.select(["date", "ticker", "sector"] + SENT_COLS)

# Left join по (ticker, date), затем forward-fill в пределах тикера (до 7 строк)
dataset = prices.join(sent_cols, on=["ticker", "date"], how="left")
dataset = dataset.sort(["ticker", "date"]).with_columns([
    pl.col(c).forward_fill(limit=7).over("ticker") for c in SENT_COLS + ["sector"]
])

# Добавляем sector отдельно (может быть null если нет новостей за 7 дней)
# sector берём из tickers.json через features
ticker_sector = features.select(["ticker", "sector"]).unique()
dataset = dataset.drop("sector").join(ticker_sector, on="ticker", how="left")

# Убираем строки без target (последний день тикера) и без цены
dataset = dataset.filter(
    pl.col("target").is_not_null() &
    pl.col("close").is_not_null()
)

# Заполняем NaN в sentiment нулями (если нет новостей за 7 дней)
dataset = dataset.with_columns([
    pl.col(c).fill_null(0.0) for c in SENT_COLS
])

print(f"\nДатасет: {len(dataset):,} строк")
print(f"Тикеров: {dataset['ticker'].n_unique()}")
print(f"Период:  {dataset['date'].min()} → {dataset['date'].max()}")
print(f"NaN в target: {dataset['target'].is_null().sum()}")
print(f"Строк с сентиментом != 0: {dataset.filter(pl.col('ticker_sent_1d') != 0).shape[0]:,}")

# ─── Сохранение ──────────────────────────────────────────────────────────────

dataset.write_parquet(OUTPUT_PARQUET)
print(f"\nСохранено → {OUTPUT_PARQUET}")
print(f"Колонки: {dataset.columns}")
