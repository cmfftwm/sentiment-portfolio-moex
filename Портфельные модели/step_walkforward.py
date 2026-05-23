"""
step_walkforward.py

Walk-forward validation: переобучаем LightGBM каждый квартал,
предсказываем только следующий квартал (честный OOS бэктест).

Схема фолдов (квартальное переобучение):
  Фолд 1:  Train 2019–2020  → Test Q1 2021
  Фолд 2:  Train 2019–Q1 2021 → Test Q2 2021
  ...
  Фолд N:  Train 2019–Qk    → Test Q(k+1)

Каждый день оценивается моделью, которая НЕ видела будущего.

Вход:  dataset.parquet
Выход: equity_curve_wf.csv + backtest_report_wf.txt + wf_fold_models/
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PARQUET = os.path.join(BASE_DIR, "dataset_vol.parquet")
EQUITY_PATH   = os.path.join(BASE_DIR, "equity_curve_wf.csv")
REPORT_PATH   = os.path.join(BASE_DIR, "backtest_report_wf.txt")
MODELS_DIR    = os.path.join(BASE_DIR, "wf_fold_models")
CBR_CSV       = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "cbr_key_rate_2020_2025.csv")
IMOEX_DB      = os.path.join(BASE_DIR, "..", "..", "Парсер рыночных данных", "moex_indices.db")

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("Walk-Forward Validation (квартальное переобучение)")
print("=" * 60)

# ─── Параметры ───────────────────────────────────────────────────────────────

TOP_N             = 5
MARKET_PERCENTILE = 20
COMMISSION        = 0.0003
MIN_TRAIN_END     = "2020-12-31"   # минимум 2 года для первого обучения
TEST_START        = "2021-01-01"   # начало тестового периода

FEATURE_COLS = [
    "ticker_sent_1d", "ticker_sent_7d", "ticker_sent_14d", "ticker_sent_mom",
    "ticker_news_count_1d", "ticker_news_count_7d",
    "market_sent_1d", "market_sent_7d",
    "sector_sent_1d", "sector_sent_7d",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_63d",
    "vol_10d", "vol_20d",
    "rsi_14", "ma_ratio_5_20", "ma_ratio_20_60", "bb_pos",
    "volume_ratio_5d", "volume_ratio_20d",
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

# ─── Загрузка данных ─────────────────────────────────────────────────────────

df = pl.read_parquet(INPUT_PARQUET).to_pandas()
df["date"] = pd.to_datetime(df["date"])
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
df["target"]    = df["target"].fillna(0)
df["target_5d"] = df["target_5d"].fillna(np.nan)

cbr_raw = pd.read_csv(CBR_CSV, parse_dates=["date"])
cbr_raw["daily_ret"] = (1 + cbr_raw["key_rate"] / 100) ** (1 / 365) - 1
cbr_rates = cbr_raw.set_index("date")["daily_ret"].to_dict()

print(f"Данных: {len(df):,} строк, тикеров: {df['ticker'].nunique()}")
print(f"Период: {df['date'].min().date()} → {df['date'].max().date()}")

# ─── Определяем квартальные фолды ─────────────────────────────────────────────

# Начало каждого тестового квартала
test_quarters = pd.date_range(start=TEST_START, end=df["date"].max(), freq="QS")

print(f"\nФолдов (кварталов): {len(test_quarters)}")
for i, q in enumerate(test_quarters):
    q_end = q + pd.offsets.QuarterEnd()
    print(f"  Фолд {i+1:2d}: Train до {(q - pd.Timedelta(days=1)).date()} → Test {q.date()} – {min(q_end, df['date'].max()).date()}")

# ─── Walk-Forward: обучение и предсказание ───────────────────────────────────

all_predictions = []  # (date, ticker, lgb_pred)

for fold_idx, test_start in enumerate(test_quarters):
    test_end = test_start + pd.offsets.QuarterEnd()
    train_end = test_start - pd.Timedelta(days=1)

    train = df[df["date"] <= train_end].copy()
    test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    # Исключаем последние 5 торговых дней: их target_5d захватывает тестовый квартал
    trade_dates = sorted(df[df["date"] < test_start]["date"].unique())
    leak_cutoff = trade_dates[-5] if len(trade_dates) >= 5 else pd.Timestamp("2000-01-01")
    train_fit = train[(train["target_5d"].notna()) & (train["date"] < leak_cutoff)]

    if len(train_fit) < 1000:
        print(f"  Фолд {fold_idx+1}: мало данных для обучения ({len(train_fit)} строк), пропуск")
        continue
    if len(test) == 0:
        continue

    X_train = train_fit[FEATURE_COLS]
    y_train = train_fit["target_5d"]

    # Валидация — последние 10% обучения
    val_cut = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_tr, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)],
    )

    # Сохраняем модель фолда
    model_path = os.path.join(MODELS_DIR, f"model_fold_{fold_idx+1:02d}.lgb")
    model.save_model(model_path)

    # Предсказания на тестовом квартале
    test["lgb_pred"] = model.predict(test[FEATURE_COLS])

    preds = test[["date", "ticker", "lgb_pred", "target", "market_sent_7d", "ticker_sent_7d"]].copy()
    all_predictions.append(preds)

    print(f"  Фолд {fold_idx+1:2d}: train={len(train_fit):,} строк, test={len(test):,}, iter={model.best_iteration}, "
          f"IC={np.corrcoef(model.predict(test[FEATURE_COLS]), test['target_5d'].fillna(0))[0,1]:.4f}")

# ─── Сборка предсказаний ──────────────────────────────────────────────────────

pred_df = pd.concat(all_predictions, ignore_index=True)
pred_df["date"] = pd.to_datetime(pred_df["date"])
print(f"\nПредсказания собраны: {len(pred_df):,} строк")
print(f"Период: {pred_df['date'].min().date()} → {pred_df['date'].max().date()}")

# ─── Бэктест на walk-forward предсказаниях ───────────────────────────────────

# Expanding-окно для порога выхода в кэш: порог рассчитывается только
# по истории до текущего дня (без look-ahead). Первые MIN_HIST_DAYS дней
# защитный фильтр неактивен из-за недостатка данных для устойчивого перцентиля.
MIN_HIST_DAYS = 60
dates = sorted(pred_df["date"].unique())
daily_market = pred_df.groupby("date")["market_sent_7d"].mean().to_dict()
sent_history = []

returns_wf = []
returns_v1 = []
prev_wf, prev_v1 = set(), set()

for date in dates:
    if len(sent_history) >= MIN_HIST_DAYS:
        market_threshold_wf = np.percentile(sent_history, MARKET_PERCENTILE)
    else:
        market_threshold_wf = -np.inf
    sent_history.append(daily_market[date])

    day = pred_df[pred_df["date"] == date]
    market_sent = day["market_sent_7d"].mean()
    cbr_daily   = cbr_rates.get(date, 0.0)

    if market_sent < market_threshold_wf:
        returns_wf.append({"date": date, "ret": cbr_daily, "in_market": False})
        returns_v1.append({"date": date, "ret": cbr_daily, "in_market": False})
        continue

    # WF: по lgb_pred
    top_wf = day.sort_values("lgb_pred", ascending=False).head(TOP_N)
    if len(top_wf) > 0:
        cur = set(top_wf["ticker"])
        to  = len(cur.symmetric_difference(prev_wf)) / max(len(cur), 1)
        returns_wf.append({"date": date, "ret": top_wf["target"].mean() - to * COMMISSION, "in_market": True})
        prev_wf = cur
    else:
        returns_wf.append({"date": date, "ret": 0.0, "in_market": False})

    # V1 baseline: по ticker_sent_7d
    top_v1 = day.sort_values("ticker_sent_7d", ascending=False).head(10)
    if len(top_v1) > 0:
        cur = set(top_v1["ticker"])
        to  = len(cur.symmetric_difference(prev_v1)) / max(len(cur), 1)
        returns_v1.append({"date": date, "ret": top_v1["target"].mean() - to * COMMISSION, "in_market": True})
        prev_v1 = cur
    else:
        returns_v1.append({"date": date, "ret": 0.0, "in_market": False})

# ─── Метрики ─────────────────────────────────────────────────────────────────

df_wf = pd.DataFrame(returns_wf).set_index("date")
df_v1 = pd.DataFrame(returns_v1).set_index("date")
df_wf["equity"] = (1 + df_wf["ret"]).cumprod()
df_v1["equity"] = (1 + df_v1["ret"]).cumprod()

def calc_metrics(rets, label):
    eq = (1 + rets).cumprod()
    ar = eq.iloc[-1] ** (252 / len(rets)) - 1
    av = rets.std() * np.sqrt(252)
    return {
        "label":     label,
        "total":     eq.iloc[-1] - 1,
        "annual":    ar,
        "vol":       av,
        "sharpe":    ar / av if av > 0 else 0,
        "max_dd":    (eq / eq.cummax() - 1).min(),
    }

m_wf = calc_metrics(df_wf["ret"], f"WF LightGBM-5d (TOP{TOP_N}, p{MARKET_PERCENTILE})")
m_v1 = calc_metrics(df_v1["ret"], "v1 сентимент (TOP10, p25)")

cbr_s  = pd.Series([cbr_rates.get(d, 0.) for d in df_wf.index], index=df_wf.index)
cbr_eq = (1 + cbr_s).cumprod()
m_cbr  = calc_metrics(cbr_s, "ЦБ")

imoex_eq, m_imoex = None, None
if os.path.exists(IMOEX_DB):
    con = sqlite3.connect(IMOEX_DB)
    im  = pd.read_sql("SELECT time,close FROM imoex ORDER BY time", con, parse_dates=["time"]); con.close()
    im["date"] = im["time"].dt.normalize()
    im = im.groupby("date")["close"].last().reset_index()
    im = im[im["date"] >= pd.Timestamp(TEST_START)]
    if len(im) > 0:
        im["ret"] = im["close"].pct_change().fillna(0)
        im["equity"] = (1 + im["ret"]).cumprod()
        imoex_eq = im.set_index("date")["equity"]
        m_imoex = calc_metrics(im["ret"], "IMOEX")

# ─── Вывод ───────────────────────────────────────────────────────────────────

print(f"\n{'─' * 80}")
print(f"{'Метрика':<25} {'WF LightGBM-5d':>18} {'v1 сентимент':>14} {'ЦБ':>8}" +
      (f" {'IMOEX':>8}" if m_imoex else ""))
print(f"{'─' * 80}")
for key, label in [("total","Доходность итого"),("annual","Доходность годовая"),
                    ("vol","Волатильность год."),("sharpe","Sharpe ratio"),("max_dd","Max Drawdown")]:
    fmt = ".3f" if key == "sharpe" else ".1%"
    row = f"{label:<25} {m_wf[key]:>17{fmt}} {m_v1[key]:>13{fmt}} {m_cbr[key]:>7{fmt}}"
    if m_imoex: row += f" {m_imoex[key]:>7{fmt}}"
    print(row)
print(f"{'─' * 80}")

cash_days = df_wf[~df_wf["in_market"]].shape[0]
print(f"\nДней в кэше: {cash_days} ({100*cash_days/len(df_wf):.1f}%)")
print(f"Фолдов обучено: {len(test_quarters)}")

# ─── Сохранение ──────────────────────────────────────────────────────────────

equity_out = pd.DataFrame({
    "date": df_wf.index,
    "wf":   df_wf["equity"].values,
    "v1":   df_v1["equity"].values,
    "cbr":  cbr_eq.reindex(df_wf.index).values,
})
if imoex_eq is not None:
    equity_out["imoex"] = imoex_eq.reindex(df_wf.index).values
equity_out.to_csv(EQUITY_PATH, index=False)
print(f"\nEquity curve → {EQUITY_PATH}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("Walk-Forward Backtest Report\n")
    f.write("=" * 55 + "\n")
    f.write(f"Переобучение: каждый квартал\n")
    f.write(f"TOP_N={TOP_N}, p{MARKET_PERCENTILE}, COMMISSION={COMMISSION}\n\n")
    for m in [m_wf, m_v1, m_cbr] + ([m_imoex] if m_imoex else []):
        f.write(f"\n[{m['label']}]\n")
        f.write(f"  Доходность итого:   {m['total']:.1%}\n")
        f.write(f"  Доходность годовая: {m['annual']:.1%}\n")
        f.write(f"  Волатильность:      {m['vol']:.1%}\n")
        f.write(f"  Sharpe:             {m['sharpe']:.3f}\n")
        f.write(f"  Max Drawdown:       {m['max_dd']:.1%}\n")
print(f"Отчёт → {REPORT_PATH}")
