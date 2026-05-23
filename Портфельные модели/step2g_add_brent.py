"""
step2g_add_brent.py

Добавляет признаки нефти Brent в dataset_vol_divs.parquet:

  brent_ret_5d   — 5-дневная доходность Brent (USD)
  brent_ret_20d  — 20-дневная доходность Brent
  brent_vol_20d  — волатильность Brent за 20 дней (аннуализированная)

Вход:  dataset_vol_divs.parquet + brent_daily.csv
Выход: dataset_vol_brent.parquet
"""

import os
import pandas as pd
import polars as pl
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

INPUT_PARQUET  = os.path.join(BASE_DIR, "dataset_vol_divs.parquet")
BRENT_CSV      = os.path.join(PROJECT_DIR, "Парсер рыночных данных", "brent_daily.csv")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol_brent.parquet")

print("=" * 60)
print("Шаг 2g: Добавление признаков нефти Brent")
print("=" * 60)

# ─── Загрузка данных ─────────────────────────────────────────────────────────

base = pl.read_parquet(INPUT_PARQUET).to_pandas()
base["date"] = pd.to_datetime(base["date"])
print(f"Базовый датасет: {len(base):,} строк, {base['ticker'].nunique()} тикеров")

brent = pd.read_csv(BRENT_CSV, parse_dates=["date"]).sort_values("date")
print(f"Brent данные: {len(brent)} дней, {brent['date'].min().date()} → {brent['date'].max().date()}")

# ─── Расчёт признаков ────────────────────────────────────────────────────────

brent["brent_ret_5d"]  = brent["close"].pct_change(5)
brent["brent_ret_20d"] = brent["close"].pct_change(20)
brent["brent_vol_20d"] = brent["close"].pct_change().rolling(20).std() * np.sqrt(252)

brent_feats = brent[["date", "brent_ret_5d", "brent_ret_20d", "brent_vol_20d"]].set_index("date")

# ─── Джойн к датасету ────────────────────────────────────────────────────────

# Для каждой даты берём последнее известное значение Brent (forward-fill)
all_dates = pd.Series(base["date"].unique()).sort_values()
brent_aligned = brent_feats.reindex(all_dates).ffill()

base = base.merge(
    brent_aligned.reset_index().rename(columns={"index": "date"}),
    on="date", how="left"
)

# Заполняем оставшиеся NaN (начало ряда)
for col in ["brent_ret_5d", "brent_ret_20d", "brent_vol_20d"]:
    base[col] = base[col].fillna(0.0)

# ─── Статистика ──────────────────────────────────────────────────────────────

new_cols = ["brent_ret_5d", "brent_ret_20d", "brent_vol_20d"]
print(f"\nНовые признаки:")
for col in new_cols:
    print(f"  {col:<20} среднее={base[col].mean():.4f}, std={base[col].std():.4f}")

coverage = (base["brent_ret_20d"] != 0).mean()
print(f"\nПокрытие (ненулевых): {coverage:.1%}")

# ─── Сохранение ──────────────────────────────────────────────────────────────

dataset = pl.from_pandas(base)
dataset.write_parquet(OUTPUT_PARQUET)
print(f"\nСохранено → {OUTPUT_PARQUET}")
print(f"Всего признаков: {len(base.columns)}")
print(f"Новые: {new_cols}")
