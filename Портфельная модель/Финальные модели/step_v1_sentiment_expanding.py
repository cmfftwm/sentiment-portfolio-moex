"""
step_v1_sentiment_expanding.py

Простейшая стратегия по сентименту (v1), честная версия без look-ahead:
порог 20-го перцентиля market_sent_7d считается по expanding окну —
только на истории до текущего дня.

Правило:
  1. Каждый торговый день: если market_sent_7d < expanding percentile(20)
     по истории до этого дня → в кэш под ставку ЦБ.
  2. Иначе: top-10 тикеров по ticker_sent_7d, равновзвешенно (10% на акцию).
  3. Ежедневный ребаланс, комиссия 0.03% от turnover.

Вход:  dataset_vol.parquet
Выход: equity_curve_v1_expanding.csv
       backtest_report_v1_expanding.txt
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import polars as pl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol.parquet")
EQUITY_PATH   = os.path.join(BASE_DIR, "equity_curve_v1_expanding.csv")
REPORT_PATH   = os.path.join(BASE_DIR, "backtest_report_v1_expanding.txt")
CBR_CSV       = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "cbr_key_rate_2020_2025.csv")
IMOEX_DB      = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "moex_indices.db")

TOP_N             = 10
MARKET_PERCENTILE = 20
COMMISSION        = 0.0003
START_DATE        = "2021-01-01"
MIN_HIST_DAYS     = 60   # минимум дней истории для expanding percentile

print("=" * 60)
print("v1 (сентимент, expanding percentile — без look-ahead)")
print("=" * 60)

# ─── Данные ──────────────────────────────────────────────────────────────────

df = pl.read_parquet(INPUT_PARQUET).to_pandas()
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= START_DATE].copy()

for col in ["ticker_sent_7d", "market_sent_7d", "target"]:
    df[col] = df[col].fillna(0)

cbr_raw = pd.read_csv(CBR_CSV, parse_dates=["date"])
cbr_raw["daily_ret"] = (1 + cbr_raw["key_rate"] / 100) ** (1 / 365) - 1
cbr_rates = cbr_raw.set_index("date")["daily_ret"].to_dict()

# ─── Expanding percentile по market_sent_7d ──────────────────────────────────

daily_market = df.groupby("date")["market_sent_7d"].mean().sort_index()
dates = daily_market.index.tolist()

# На каждый день — порог по всей истории ДО этого дня (строго меньше).
history = []
thresholds = {}
for d in dates:
    if len(history) >= MIN_HIST_DAYS:
        thresholds[d] = np.percentile(history, MARKET_PERCENTILE)
    else:
        thresholds[d] = -np.inf  # пока мало истории — не уходим в кэш
    history.append(daily_market[d])

# ─── Симуляция ───────────────────────────────────────────────────────────────

returns = []
prev = set()

for d in dates:
    day = df[df["date"] == d]
    market_sent = daily_market[d]
    cbr_daily   = cbr_rates.get(d, 0.0)

    if market_sent < thresholds[d]:
        returns.append({"date": d, "ret": cbr_daily, "in_market": False})
        prev = set()
        continue

    top = day[["ticker", "ticker_sent_7d", "target"]] \
            .sort_values("ticker_sent_7d", ascending=False).head(TOP_N)
    if len(top) == 0:
        returns.append({"date": d, "ret": 0.0, "in_market": False})
        prev = set()
        continue

    cur = set(top["ticker"])
    turnover = len(cur.symmetric_difference(prev)) / max(len(cur), 1)
    ret = top["target"].mean() - turnover * COMMISSION
    returns.append({"date": d, "ret": ret, "in_market": True})
    prev = cur

# ─── Метрики ─────────────────────────────────────────────────────────────────

out = pd.DataFrame(returns).set_index("date")
out["equity"] = (1 + out["ret"]).cumprod()

def calc_metrics(rets, label):
    eq = (1 + rets).cumprod()
    ar = eq.iloc[-1] ** (252 / len(rets)) - 1
    av = rets.std() * np.sqrt(252)
    return {
        "label":  label,
        "total":  eq.iloc[-1] - 1,
        "annual": ar,
        "vol":    av,
        "sharpe": ar / av if av > 0 else 0,
        "max_dd": (eq / eq.cummax() - 1).min(),
    }

m_v1 = calc_metrics(out["ret"], "v1 expanding")

cbr_s   = pd.Series([cbr_rates.get(d, 0.0) for d in out.index], index=out.index)
cbr_eq  = (1 + cbr_s).cumprod()
m_cbr   = calc_metrics(cbr_s, "ЦБ")

m_imoex, imoex_eq = None, None
if os.path.exists(IMOEX_DB):
    con = sqlite3.connect(IMOEX_DB)
    im  = pd.read_sql("SELECT time, close FROM imoex ORDER BY time", con, parse_dates=["time"])
    con.close()
    im["date"] = im["time"].dt.normalize()
    im = im.groupby("date")["close"].last().reset_index()
    im = im[im["date"] >= pd.Timestamp(START_DATE)]
    if len(im) > 0:
        im["ret"] = im["close"].pct_change().fillna(0)
        imoex_eq = (1 + im.set_index("date")["ret"]).cumprod()
        m_imoex  = calc_metrics(im["ret"], "IMOEX")

# ─── Вывод ───────────────────────────────────────────────────────────────────

cash_days = (~out["in_market"]).sum()
print(f"\nТорговых дней: {len(out)}")
print(f"Дней в кэше:   {cash_days} ({100*cash_days/len(out):.1f}%)")

print(f"\n{'─'*70}")
print(f"{'Метрика':<25} {'v1 expanding':>15} {'ЦБ':>10}" + (f" {'IMOEX':>10}" if m_imoex else ""))
print(f"{'─'*70}")
for key, label in [("total","Доходность итого"),("annual","Доходность годовая"),
                   ("vol","Волатильность год."),("sharpe","Sharpe ratio"),("max_dd","Max Drawdown")]:
    fmt = ".3f" if key == "sharpe" else ".1%"
    row = f"{label:<25} {m_v1[key]:>14{fmt}} {m_cbr[key]:>9{fmt}}"
    if m_imoex: row += f" {m_imoex[key]:>9{fmt}}"
    print(row)
print(f"{'─'*70}")

print("\nГодовая разбивка:")
for year in range(2021, 2026):
    yr = out["equity"][out.index.year == year]
    if len(yr) < 2: continue
    print(f"  {year}: {yr.iloc[-1]/yr.iloc[0] - 1:+.1%}")

# ─── Сохранение ──────────────────────────────────────────────────────────────

equity_out = pd.DataFrame({
    "date":         out.index,
    "v1_expanding": out["equity"].values,
    "cbr":          cbr_eq.reindex(out.index).values,
})
if imoex_eq is not None:
    equity_out["imoex"] = imoex_eq.reindex(out.index).values
equity_out.to_csv(EQUITY_PATH, index=False)
print(f"\nEquity → {EQUITY_PATH}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("Backtest Report — v1 (sentiment, expanding percentile)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Правило: top-{TOP_N} по ticker_sent_7d, равновзвешенно,\n")
    f.write(f"в кэш под ставку ЦБ если market_sent_7d < expanding p{MARKET_PERCENTILE}\n")
    f.write(f"(мин. истории {MIN_HIST_DAYS} дней). Комиссия {COMMISSION:.2%}.\n\n")
    for m in [m_v1, m_cbr] + ([m_imoex] if m_imoex else []):
        f.write(f"\n[{m['label']}]\n")
        f.write(f"  Доходность итого:   {m['total']:.1%}\n")
        f.write(f"  Доходность годовая: {m['annual']:.1%}\n")
        f.write(f"  Волатильность:      {m['vol']:.1%}\n")
        f.write(f"  Sharpe:             {m['sharpe']:.3f}\n")
        f.write(f"  Max Drawdown:       {m['max_dd']:.1%}\n")
    f.write(f"\nДней в кэше: {cash_days} ({100*cash_days/len(out):.1f}%)\n")
    f.write("\nГодовая разбивка:\n")
    for year in range(2021, 2026):
        yr = out["equity"][out.index.year == year]
        if len(yr) < 2: continue
        f.write(f"  {year}: {yr.iloc[-1]/yr.iloc[0] - 1:+.1%}\n")
print(f"Отчёт → {REPORT_PATH}")
