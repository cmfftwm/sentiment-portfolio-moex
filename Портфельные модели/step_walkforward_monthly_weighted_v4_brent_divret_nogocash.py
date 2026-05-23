"""
step_walkforward_monthly_weighted_v4_brent_divret_nogocash.py

v4_brent_divret С сентиментом, БЕЗ ухода в кэш (ни go_cash, ни CBR pos_scale).
Цель: изолировать вклад сентимента ТОЛЬКО в выбор акций, без влияния на тайминг.

Вход:  dataset_vol_brent.parquet + dividends_moex.csv
Выход: equity_curve_wf_monthly_v4_brent_divret_nogocash.csv
       backtest_report_wf_monthly_v4_brent_divret_nogocash.txt
       wf_monthly_v4_brent_divret_nogocash_models/
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol_brent.parquet")
EQUITY_PATH   = os.path.join(BASE_DIR, "equity_curve_wf_monthly_v4_brent_divret_nogocash.csv")
REPORT_PATH   = os.path.join(BASE_DIR, "backtest_report_wf_monthly_v4_brent_divret_nogocash.txt")
MODELS_DIR    = os.path.join(BASE_DIR, "wf_monthly_v4_brent_divret_nogocash_models")
DIVS_CSV      = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "dividends_moex.csv")
USDRUB_CSV    = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "usdrub_cbr.csv")
CBR_CSV       = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "cbr_key_rate_2020_2025.csv")
IMOEX_DB      = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "moex_indices.db")

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("WF Monthly v4_brent_divret С сентиментом БЕЗ go_cash")
print("=" * 60)

# ─── Параметры ───────────────────────────────────────────────────────────────

COMMISSION     = 0.0003
TEST_START     = "2021-01-01"
SOFTMAX_TEMP   = 0.1
DECAY_HALFLIFE = 365

TOP_N_MIN      = 3
TOP_N_DEFAULT  = 5
TOP_N_MAX      = 7
SPREAD_HIGH    = 0.02
SPREAD_LOW     = 0.005

# go_cash и CBR pos_scale — ОТКЛЮЧЕНЫ

FEATURE_COLS = [
    "ticker_sent_1d", "ticker_sent_7d", "ticker_sent_14d", "ticker_sent_mom",
    "ticker_news_count_1d", "ticker_news_count_7d",
    "market_sent_1d", "market_sent_7d",
    "sector_sent_1d", "sector_sent_7d",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_63d",
    "vol_10d", "vol_20d",
    "rsi_14", "ma_ratio_5_20", "ma_ratio_20_60", "bb_pos",
    "volume_ratio_5d", "volume_ratio_20d",
    "cbr_rate", "imoex_ret_20d", "imoex_vol_20d",
    "usdrub", "usdrub_ret_20d",
    "ticker_neg_7d", "ticker_pos_7d",
    "market_neg_7d", "market_pos_7d",
    "div_yield", "days_to_exdate", "div_season",
    "brent_ret_5d", "brent_ret_20d", "brent_vol_20d",
]

LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "learning_rate":    0.01,
    "num_leaves":       31,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq":     5,
    "lambda_l1":        0.05,
    "lambda_l2":        0.05,
    "verbose":          -1,
    "seed":             42,
}

# ─── Загрузка данных ──────────────────────────────────────────────────────────

df = pl.read_parquet(INPUT_PARQUET).to_pandas()
df["date"] = pd.to_datetime(df["date"])

cbr_raw = pd.read_csv(CBR_CSV, parse_dates=["date"])
cbr_raw["daily_ret"] = (1 + cbr_raw["key_rate"] / 100) ** (1 / 365) - 1
cbr_rates    = cbr_raw.set_index("date")["daily_ret"].to_dict()
cbr_rate_map = cbr_raw.set_index("date")["key_rate"]

con = sqlite3.connect(IMOEX_DB)
im  = pd.read_sql("SELECT time, close FROM imoex ORDER BY time", con, parse_dates=["time"])
con.close()
im["date"] = im["time"].dt.normalize()
im = im.groupby("date")["close"].last().reset_index().sort_values("date")
im["imoex_ret_20d"] = im["close"].pct_change(20)
im["imoex_vol_20d"] = im["close"].pct_change().rolling(20).std() * np.sqrt(252)
imoex_macro = im.set_index("date")[["imoex_ret_20d", "imoex_vol_20d"]]
im_oos      = im[im["date"] >= pd.Timestamp(TEST_START)].copy()

df["cbr_rate"]      = cbr_rate_map.reindex(df["date"]).ffill().values
df["imoex_ret_20d"] = imoex_macro["imoex_ret_20d"].reindex(df["date"]).ffill().values
df["imoex_vol_20d"] = imoex_macro["imoex_vol_20d"].reindex(df["date"]).ffill().values

usdrub_df = pd.read_csv(USDRUB_CSV, parse_dates=["date"]).set_index("date")["usdrub"]
usdrub_df = usdrub_df.sort_index()
usdrub_ret20 = usdrub_df.pct_change(20)
df["usdrub"]         = usdrub_df.reindex(df["date"]).ffill().values
df["usdrub_ret_20d"] = usdrub_ret20.reindex(df["date"]).ffill().values

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
df["target"]     = df["target"].fillna(0)
df["target_5d"]  = df["target_5d"].fillna(np.nan)

# ─── Загрузка дивидендов для бэктеста ────────────────────────────────────────

divs_raw = pd.read_csv(DIVS_CSV, parse_dates=["registryclosedate"])
divs_raw = divs_raw[divs_raw["currencyid"] == "RUB"].copy()

price_on_date = df.set_index(["ticker", "date"])["close"]
div_map = {}
for _, row in divs_raw.iterrows():
    exdate = row["registryclosedate"]
    ticker = row["ticker"]
    value  = row["value"]
    try:
        price = price_on_date.loc[(ticker, exdate)]
        if price > 0:
            div_map.setdefault(exdate, {})[ticker] = value / price
    except KeyError:
        ticker_prices = df[df["ticker"] == ticker].set_index("date")["close"]
        nearby = ticker_prices[ticker_prices.index <= exdate]
        if not nearby.empty and nearby.iloc[-1] > 0:
            div_map.setdefault(nearby.index[-1], {})[ticker] = value / nearby.iloc[-1]

print(f"Данных: {len(df):,} строк, тикеров: {df['ticker'].nunique()}")
print(f"Период: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Признаков: {len(FEATURE_COLS)} (с сентиментом)")
print(f"go_cash: ОТКЛЮЧЁН | CBR pos_scale: ОТКЛЮЧЁН")

# ─── Фолды ───────────────────────────────────────────────────────────────────

test_months = pd.date_range(start=TEST_START, end=df["date"].max(), freq="MS")
print(f"Фолдов: {len(test_months)}\n")

# ─── Walk-Forward ─────────────────────────────────────────────────────────────

all_predictions = []

for fold_idx, test_start in enumerate(test_months):
    test_end  = test_start + pd.offsets.MonthEnd()
    train_end = test_start - pd.Timedelta(days=1)

    train = df[df["date"] <= train_end].copy()
    test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    trade_dates = sorted(df[df["date"] < test_start]["date"].unique())
    leak_cutoff = trade_dates[-5] if len(trade_dates) >= 5 else pd.Timestamp("2000-01-01")
    train_fit   = train[(train["target_5d"].notna()) & (train["date"] < leak_cutoff)]

    if len(train_fit) < 1000 or len(test) == 0:
        continue

    min_date      = train_fit["date"].min()
    days_elapsed  = (train_fit["date"] - min_date).dt.days.values
    sample_weight = np.exp(days_elapsed * np.log(2) / DECAY_HALFLIFE)

    X_train = train_fit[FEATURE_COLS]
    y_train = train_fit["target_5d"]
    val_cut = int(len(X_train) * 0.9)

    dtrain = lgb.Dataset(X_train.iloc[:val_cut], label=y_train.iloc[:val_cut],
                         weight=sample_weight[:val_cut])
    dval   = lgb.Dataset(X_train.iloc[val_cut:], label=y_train.iloc[val_cut:],
                         reference=dtrain)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)],
    )
    model.save_model(os.path.join(MODELS_DIR, f"model_fold_{fold_idx+1:03d}.lgb"))

    test["lgb_pred"] = model.predict(test[FEATURE_COLS])
    preds = test[["date", "ticker", "lgb_pred", "target"]].copy()
    all_predictions.append(preds)

    ic = np.corrcoef(test["lgb_pred"], test["target_5d"].fillna(0))[0, 1]
    print(f"  Фолд {fold_idx+1:2d} ({test_start.strftime('%Y-%m')}): "
          f"train={len(train_fit):,}, iter={model.best_iteration}, IC={ic:.4f}")

# ─── Бэктест ─────────────────────────────────────────────────────────────────

pred_df = pd.concat(all_predictions, ignore_index=True)
pred_df["date"] = pd.to_datetime(pred_df["date"])
print(f"\nПредсказаний: {len(pred_df):,} | {pred_df['date'].min().date()} → {pred_df['date'].max().date()}")

def softmax_weights(scores, temp=SOFTMAX_TEMP):
    s = np.array(scores, dtype=float) / temp
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()

def dynamic_top_n(scores):
    arr = np.array(scores)
    spread = arr.max() - np.median(arr)
    if spread >= SPREAD_HIGH:
        return TOP_N_MIN
    elif spread <= SPREAD_LOW:
        return TOP_N_MAX
    return TOP_N_DEFAULT

dates = sorted(pred_df["date"].unique())

returns_w  = []
returns_eq = []
prev_w, prev_eq = {}, set()

for date in dates:
    day = pred_df[pred_df["date"] == date]

    # go_cash ОТКЛЮЧЁН — всегда в рынке
    all_scores = day["lgb_pred"].values
    n   = dynamic_top_n(all_scores)
    top = day.sort_values("lgb_pred", ascending=False).head(n)

    if len(top) == 0:
        returns_w.append( {"date": date, "ret": 0.0, "in_market": False, "div_ret": 0.0})
        returns_eq.append({"date": date, "ret": 0.0, "in_market": False})
        continue

    # pos_scale = 1.0 всегда
    weights  = softmax_weights(top["lgb_pred"].values)
    cur_w    = dict(zip(top["ticker"], weights))
    all_t    = set(cur_w) | set(prev_w)
    to_w     = sum(abs(cur_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_t) / 2
    port_ret = (top["target"].values * weights).sum() - to_w * COMMISSION

    todays_divs = div_map.get(date, {})
    div_ret_w   = sum(cur_w.get(t, 0.0) * dy for t, dy in todays_divs.items())
    port_ret   += div_ret_w

    returns_w.append({"date": date, "ret": port_ret, "in_market": True, "div_ret": div_ret_w})
    prev_w = cur_w

    cur_eq  = set(top["ticker"])
    to_eq   = len(cur_eq.symmetric_difference(prev_eq)) / max(len(cur_eq), 1)
    eq_ret  = top["target"].mean() - to_eq * COMMISSION
    div_ret_eq = sum((1/len(cur_eq)) * dy for t, dy in todays_divs.items() if t in cur_eq)
    eq_ret += div_ret_eq
    returns_eq.append({"date": date, "ret": eq_ret, "in_market": True})
    prev_eq = cur_eq

# ─── Метрики ─────────────────────────────────────────────────────────────────

df_w  = pd.DataFrame(returns_w).set_index("date")
df_eq = pd.DataFrame(returns_eq).set_index("date")
df_w["equity"]  = (1 + df_w["ret"]).cumprod()
df_eq["equity"] = (1 + df_eq["ret"]).cumprod()

def calc_metrics(rets, label):
    eq = (1 + rets).cumprod()
    ar = eq.iloc[-1] ** (252 / len(rets)) - 1
    av = rets.std() * np.sqrt(252)
    return {"label": label, "total": eq.iloc[-1] - 1, "annual": ar,
            "vol": av, "sharpe": ar / av if av > 0 else 0,
            "max_dd": (eq / eq.cummax() - 1).min()}

cbr_s = pd.Series([cbr_rates.get(d, 0.) for d in df_w.index], index=df_w.index)
m_w   = calc_metrics(df_w["ret"],  "sent_nogocash средневзв.")
m_eq  = calc_metrics(df_eq["ret"], "sent_nogocash равновзв.")
m_cbr = calc_metrics(cbr_s, "ЦБ")

imoex_eq, m_imoex = None, None
if len(im_oos) > 0:
    im_oos = im_oos.copy()
    im_oos["ret"] = im_oos["close"].pct_change().fillna(0)
    imoex_eq = (1 + im_oos.set_index("date")["ret"]).cumprod()
    m_imoex  = calc_metrics(im_oos["ret"], "IMOEX")

# ─── Вывод ───────────────────────────────────────────────────────────────────

total_div  = df_w["div_ret"].sum()
annual_div = total_div / (len(df_w) / 252)
print(f"\nДивидендный доход (средневзвеш.):")
print(f"  Всего за период:  {total_div:.2%}")
print(f"  В год (среднее):  {annual_div:.2%}")

print(f"\n{'─'*80}")
print(f"{'Метрика':<25} {'sent средневзв.':>18} {'sent равновзв.':>16} {'ЦБ':>8}" +
      (f" {'IMOEX':>8}" if m_imoex else ""))
print(f"{'─'*80}")
for key, label in [("total","Доходность итого"),("annual","Доходность годовая"),
                   ("vol","Волатильность год."),("sharpe","Sharpe ratio"),("max_dd","Max Drawdown")]:
    fmt = ".3f" if key == "sharpe" else ".1%"
    row = f"{label:<25} {m_w[key]:>15{fmt}} {m_eq[key]:>13{fmt}} {m_cbr[key]:>7{fmt}}"
    if m_imoex: row += f" {m_imoex[key]:>7{fmt}}"
    print(row)
print(f"{'─'*80}")

print(f"\nДней в рынке: 100.0% (go_cash отключён)")
print(f"Фолдов: {len(all_predictions)}")

print("\nГодовая разбивка (средневзвеш.):")
for year in range(2021, 2026):
    s  = df_w["equity"]
    yr = s[s.index.year == year]
    if len(yr) < 2: continue
    ret = yr.iloc[-1] / yr.iloc[0] - 1
    print(f"  {year}: {ret:+.1%}")

print(f"\n{'─'*50}")
print(f"\nДля сравнения:")
print(f"  v4_brent_divret (с go_cash): Sharpe 2.314 | CAGR +54.3% | Max DD -25.9%")
print(f"{'─'*50}")

# ─── Сохранение ──────────────────────────────────────────────────────────────

cbr_eq_s  = (1 + cbr_s).cumprod()
equity_out = pd.DataFrame({
    "date":          df_w.index,
    "conf_weighted": df_w["equity"].values,
    "equal_weight":  df_eq["equity"].values,
    "cbr":           cbr_eq_s.reindex(df_w.index).values,
})
if imoex_eq is not None:
    equity_out["imoex"] = imoex_eq.reindex(df_w.index).values
equity_out.to_csv(EQUITY_PATH, index=False)
print(f"\nEquity → {EQUITY_PATH}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("Walk-Forward Backtest Report (v4_brent_divret С сентиментом БЕЗ go_cash)\n")
    f.write("=" * 60 + "\n")
    f.write("go_cash ОТКЛЮЧЁН, CBR pos_scale ОТКЛЮЧЁН.\n")
    f.write(f"Признаков: {len(FEATURE_COLS)} (с сентиментом)\n\n")
    for m in [m_w, m_eq, m_cbr] + ([m_imoex] if m_imoex else []):
        f.write(f"\n[{m['label']}]\n")
        f.write(f"  Доходность итого:   {m['total']:.1%}\n")
        f.write(f"  Доходность годовая: {m['annual']:.1%}\n")
        f.write(f"  Волатильность:      {m['vol']:.1%}\n")
        f.write(f"  Sharpe:             {m['sharpe']:.3f}\n")
        f.write(f"  Max Drawdown:       {m['max_dd']:.1%}\n")
    f.write("\nГодовая разбивка (средневзвеш.):\n")
    for year in range(2021, 2026):
        s  = df_w["equity"]
        yr = s[s.index.year == year]
        if len(yr) < 2: continue
        ret = yr.iloc[-1] / yr.iloc[0] - 1
        f.write(f"  {year}: {ret:+.1%}\n")
print(f"Отчёт → {REPORT_PATH}")
