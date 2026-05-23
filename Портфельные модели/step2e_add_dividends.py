"""
step2e_add_dividends.py

Добавляет дивидендные признаки в dataset_vol_negpos.parquet:

  div_yield      — сумма дивидендов за последние 12 мес / текущая цена
  days_to_exdate — дней до ближайшей даты отсечки (252 если нет в горизонте)
  div_season     — 1 если до ближайшей отсечки <= 30 дней, иначе 0

Вход:  dataset_vol_negpos.parquet + dividends_moex.csv
Выход: dataset_vol_divs.parquet
"""

import os
import pandas as pd
import polars as pl
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

INPUT_PARQUET  = os.path.join(BASE_DIR, "dataset_vol_negpos.parquet")
DIVS_CSV       = os.path.join(PROJECT_DIR, "Парсер рыночных данных", "dividends_moex.csv")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol_divs.parquet")

print("=" * 60)
print("Шаг 2e: Добавление дивидендных признаков")
print("=" * 60)

# ─── Загрузка данных ──────────────────────────────────────────────────────────

base = pl.read_parquet(INPUT_PARQUET).to_pandas()
base["date"] = pd.to_datetime(base["date"])
print(f"Базовый датасет: {len(base):,} строк, {base['ticker'].nunique()} тикеров")

divs = pd.read_csv(DIVS_CSV, parse_dates=["registryclosedate"])
# Оставляем только рублёвые дивиденды
divs = divs[divs["currencyid"] == "RUB"].copy()
divs = divs.sort_values(["ticker", "registryclosedate"]).reset_index(drop=True)
print(f"Дивидендов: {len(divs):,} по {divs['ticker'].nunique()} тикерам")

# ─── Расчёт признаков по каждой строке датасета ───────────────────────────────

print("\nСчитаю дивидендные признаки...")

div_yield_list    = []
days_to_exdate_list = []
div_season_list   = []

# Группируем дивиденды по тикеру для быстрого доступа
divs_by_ticker = {ticker: grp for ticker, grp in divs.groupby("ticker")}

for _, row in base.iterrows():
    ticker = row["ticker"]
    date   = row["date"]
    price  = row["close"]

    ticker_divs = divs_by_ticker.get(ticker)

    if ticker_divs is None or price <= 0:
        div_yield_list.append(0.0)
        days_to_exdate_list.append(252)
        div_season_list.append(0)
        continue

    # div_yield: сумма дивидендов за последние 12 месяцев / цена
    window_start = date - pd.Timedelta(days=365)
    past_divs = ticker_divs[
        (ticker_divs["registryclosedate"] > window_start) &
        (ticker_divs["registryclosedate"] <= date)
    ]["value"].sum()
    div_yield_list.append(past_divs / price if price > 0 else 0.0)

    # days_to_exdate: дней до следующей отсечки
    future_divs = ticker_divs[ticker_divs["registryclosedate"] > date]
    if len(future_divs) > 0:
        next_exdate = future_divs["registryclosedate"].min()
        days = (next_exdate - date).days
        days_to_exdate_list.append(min(days, 252))
    else:
        days_to_exdate_list.append(252)

    # div_season: 1 если до отсечки <= 30 дней
    last_days = days_to_exdate_list[-1]
    div_season_list.append(1 if last_days <= 30 else 0)

base["div_yield"]      = div_yield_list
base["days_to_exdate"] = days_to_exdate_list
base["div_season"]     = div_season_list

# ─── Статистика ──────────────────────────────────────────────────────────────

print(f"\nСтатистика:")
print(f"  div_yield:      среднее={base['div_yield'].mean():.4f}, "
      f"макс={base['div_yield'].max():.4f}, "
      f"ненулевых={( base['div_yield'] > 0).mean():.1%}")
print(f"  days_to_exdate: среднее={base['days_to_exdate'].mean():.0f} дней, "
      f"мин={base['days_to_exdate'].min()}")
print(f"  div_season:     доля=1: {base['div_season'].mean():.1%}")

# Топ тикеров по дивидендной доходности
top_div = base.groupby("ticker")["div_yield"].mean().sort_values(ascending=False).head(10)
print(f"\nТоп тикеров по средней div_yield:")
for t, v in top_div.items():
    print(f"  {t:<8} {v:.3f} ({v*100:.1f}%)")

# ─── Сохранение ──────────────────────────────────────────────────────────────

dataset = pl.from_pandas(base)
dataset.write_parquet(OUTPUT_PARQUET)
print(f"\nСохранено → {OUTPUT_PARQUET}")
print(f"Новые признаки: div_yield, days_to_exdate, div_season")
