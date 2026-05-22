"""
fetch_brent.py

Загружает дневные цены нефти Brent (USD) через Yahoo Finance API.
Тикер: BZ=F (Brent Crude Oil Futures)

Выход: brent_daily.csv
  Колонки: date, close
"""

import os
import ssl
import json
import urllib.request
import pandas as pd

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV  = os.path.join(BASE_DIR, "brent_daily.csv")

print("=" * 60)
print("Загрузка цен Brent (Yahoo Finance, BZ=F)")
print("=" * 60)

url = "https://query1.finance.yahoo.com/v8/finance/chart/BZ=F?interval=1d&range=10y"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
    data = json.loads(r.read())

result    = data["chart"]["result"][0]
timestamps = result["timestamp"]
closes     = result["indicators"]["quote"][0]["close"]

df = pd.DataFrame({"ts": timestamps, "close": closes})
df["date"] = pd.to_datetime(df["ts"], unit="s").dt.normalize()
df = df.dropna(subset=["close"])
df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
df = df[df["date"] >= "2018-01-01"]

print(f"Период: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Строк:  {len(df)}")
print(f"\nПоследние значения:")
print(df.tail(5).to_string(index=False))

df.to_csv(OUT_CSV, index=False)
print(f"\nСохранено → {OUT_CSV}")
