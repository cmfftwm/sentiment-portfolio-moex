"""
fetch_dividends_moex.py

Загружает историю дивидендов по всем тикерам из tickers.json
через MOEX ISS API:
  https://iss.moex.com/iss/securities/{ticker}/dividends.json

Выход: dividends_moex.csv
  Колонки: ticker, registryclosedate, value, currencyid
"""

import json
import os
import ssl
import time
import urllib.request
import pandas as pd

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

TICKERS_JSON = os.path.join(PROJECT_DIR, "tickers.json")
OUT_CSV      = os.path.join(BASE_DIR, "dividends_moex.csv")

print("=" * 60)
print("Загрузка дивидендов с MOEX ISS API")
print("=" * 60)

# ─── Загрузка списка тикеров ──────────────────────────────────────────────────

with open(TICKERS_JSON, encoding="utf-8") as f:
    tickers_data = json.load(f)

tickers = [item.get("tiker") or item.get("ticker") for item in tickers_data]
tickers = [t for t in tickers if t]
print(f"Тикеров: {len(tickers)}")

# ─── Загрузка дивидендов ──────────────────────────────────────────────────────

all_rows = []
errors   = []

for i, ticker in enumerate(tickers):
    url = f"https://iss.moex.com/iss/securities/{ticker}/dividends.json"
    try:
        with urllib.request.urlopen(url, timeout=10, context=ctx) as r:
            data = json.loads(r.read())

        cols = data["dividends"]["columns"]
        rows = data["dividends"]["data"]

        if not rows:
            print(f"  {ticker}: нет данных")
            continue

        df_t = pd.DataFrame(rows, columns=cols)
        df_t["ticker"] = ticker
        all_rows.append(df_t)
        print(f"  {ticker}: {len(df_t)} записей, "
              f"последний: {df_t['registryclosedate'].max()}, "
              f"размер: {df_t['value'].max():.2f} {df_t['currencyid'].iloc[0]}")

    except Exception as e:
        errors.append(ticker)
        print(f"  {ticker}: ошибка — {e}")

    # Пауза чтобы не перегружать API
    time.sleep(0.3)

# ─── Сборка и сохранение ─────────────────────────────────────────────────────

if not all_rows:
    print("\nНет данных для сохранения")
    exit(1)

df = pd.concat(all_rows, ignore_index=True)

# Оставляем только нужные колонки
keep_cols = ["ticker", "registryclosedate", "value", "currencyid"]
df = df[[c for c in keep_cols if c in df.columns]]
df["registryclosedate"] = pd.to_datetime(df["registryclosedate"])
df = df.sort_values(["ticker", "registryclosedate"]).reset_index(drop=True)

print(f"\n{'='*60}")
print(f"Итого записей:  {len(df):,}")
print(f"Тикеров:        {df['ticker'].nunique()}")
print(f"Период:         {df['registryclosedate'].min().date()} → {df['registryclosedate'].max().date()}")
print(f"Ошибок:         {len(errors)}: {errors}")

print(f"\nПримеры:")
print(df.groupby("ticker").tail(1).head(15).to_string(index=False))

df.to_csv(OUT_CSV, index=False)
print(f"\nСохранено → {OUT_CSV}")
