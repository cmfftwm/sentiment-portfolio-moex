"""
gipotiz_h1.py

Гипотеза 1: Сентимент из Telegram-каналов опережает новостные СМИ
по влиянию на доходности акций.

Методы проверки:
  1. Information Coefficient (IC) — корреляция сентимента с будущей доходностью
     при горизонтах 5m, 1h, 4h, 1d. Сравниваем IC(TG) vs IC(СМИ).
  2. Lead-lag анализ — для совпадающих событий (тот же тикер, то же окно)
     вычисляем медианный лаг: t_сми - t_tg. Тест H0: лаг = 0.
  3. Horse-race регрессия — ret_h ~ sent_tg + sent_news.
     Если β_tg > β_news и значим → TG информативнее.
  4. Пошаговое добавление источников — IC только TG vs только СМИ vs оба вместе.
"""

import sqlite3
import sys
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr, ttest_1samp, ttest_ind
import statsmodels.formula.api as smf
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Пути
# ──────────────────────────────────────────────────────────────

TG_DB_PATH      = "telegram_messages.db"
NEWS_CSV_PATH   = "Новостные ленты/Сентимент новости/news_sentiment.csv"
PRICES_DB_PATH  = "moex_1m.db"
RESULTS_DB_PATH = "h1_results.db"
RESULTS_TXT     = f"h1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

HORIZONS = ["5m", "1h", "4h", "1d"]


# ──────────────────────────────────────────────────────────────
# 1. Загрузка данных
# ──────────────────────────────────────────────────────────────

def load_tg_sentiment() -> pl.DataFrame:
    """Загружает TG-сентимент из news_ticker_sentiment."""
    print("Загружаю Telegram-сентимент...")
    conn = sqlite3.connect(TG_DB_PATH)
    df = pl.read_database("""
        SELECT ticker, date AS event_time, sentiment
        FROM news_ticker_sentiment
        WHERE sentiment IS NOT NULL
    """, conn)
    conn.close()
    event_time_list = df["event_time"].to_list()
    df = df.with_columns(
        pl.Series("event_time",
                  pl.Series(event_time_list).str.to_datetime(strict=False, time_unit="ns"))
          .dt.replace_time_zone(None)
    )
    df = df.with_columns(pl.lit("telegram").alias("source"))
    print(f"   TG: {len(df):,} записей, тикеров: {df['ticker'].n_unique()}")
    return df


def load_news_sentiment() -> pl.DataFrame:
    """Загружает сентимент из news_sentiment.csv."""
    print("Загружаю сентимент СМИ...")
    df = pl.read_csv(NEWS_CSV_PATH, infer_schema_length=10000)
    df = df.rename({"publication_datetime": "event_time"})
    event_time_list = df["event_time"].to_list()
    df = df.with_columns(
        pl.Series("event_time",
                  pl.Series(event_time_list).str.to_datetime(strict=False, time_unit="ns"))
          .dt.replace_time_zone(None)
    )
    df = df.select(["source", "ticker", "event_time", "sentiment"])
    print(f"   СМИ: {len(df):,} записей, тикеров: {df['ticker'].n_unique()}")
    return df


def load_prices(tickers: list[str]) -> pl.DataFrame:
    """Загружает 1-минутные цены для нужных тикеров из moex_1m.db."""
    print(f"Загружаю цены для {len(tickers)} тикеров...")
    conn = sqlite3.connect(PRICES_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    available = {r[0].lower(): r[0] for r in cursor.fetchall()}

    prices_list = []
    for ticker in tqdm(tickers, desc="   Цены"):
        table = available.get(ticker.lower())
        if not table:
            continue
        try:
            chunk = pl.read_database(
                f'SELECT \'{ticker}\' AS ticker, time AS ts, close AS price '
                f'FROM "{table}" WHERE close IS NOT NULL ORDER BY time',
                conn
            )
            if len(chunk) > 0:
                prices_list.append(chunk)
        except Exception as e:
            print(f"   {ticker}: {e}")

    conn.close()
    if not prices_list:
        return pl.DataFrame({"ticker": [], "ts": [], "price": []})
    prices = pl.concat(prices_list)
    prices = prices.with_columns(
        pl.col("ts").str.to_datetime(strict=False, time_unit="ns").alias("ts")
    )
    print(f"   Загружено {len(prices):,} ценовых точек")
    return prices


# ──────────────────────────────────────────────────────────────
# 2. Event-study датасет (из gipotiz.py)
# ──────────────────────────────────────────────────────────────

def parse_horizon(horizon: str):
    h = horizon.lower().strip()
    if h.endswith("m"):
        return pl.duration(minutes=int(h[:-1]))
    elif h.endswith("h"):
        return pl.duration(hours=int(h[:-1]))
    elif h.endswith("d"):
        return pl.duration(days=int(h[:-1]))
    raise ValueError(f"Неизвестный горизонт: {horizon}")


def build_event_study(news_df: pl.DataFrame, prices: pl.DataFrame, horizon: str) -> pl.DataFrame:
    """
    Строит event-study: ticker, event_time, source, sentiment, ret_h.
    """
    news = news_df.select(["ticker", "event_time", "source", "sentiment"]).sort(["ticker", "event_time"])
    prices_s = prices.sort(["ticker", "ts"])

    # Цена в момент новости
    ev = news.join_asof(
        prices_s,
        left_on="event_time", right_on="ts",
        by="ticker", strategy="backward"
    ).rename({"price": "price_t"})

    # Цена через horizon
    dur = parse_horizon(horizon)
    ev = ev.with_columns(
        (pl.col("event_time") + dur).cast(pl.Datetime(time_unit="ns")).alias("event_time_h")
    )
    ev = ev.join_asof(
        prices_s.rename({"ts": "ts_h", "price": "price_h"}),
        left_on="event_time_h", right_on="ts_h",
        by="ticker", strategy="backward"
    )

    ev = (
        ev.filter(pl.col("price_t").is_not_null() & pl.col("price_h").is_not_null())
          .with_columns(
              ((pl.col("price_h") - pl.col("price_t")) / pl.col("price_t")).alias("ret_h")
          )
    )
    return ev


# ──────────────────────────────────────────────────────────────
# 3. Information Coefficient (IC)
# ──────────────────────────────────────────────────────────────

def compute_ic(ev: pl.DataFrame, source_name: str) -> dict:
    """Spearman-корреляция сентимента с будущей доходностью."""
    df = ev.select(["sentiment", "ret_h"]).drop_nulls().to_pandas()
    df = df[df["sentiment"] != 0]  # исключаем нейтральные
    if len(df) < 30:
        return {"source": source_name, "ic": np.nan, "p_value": np.nan, "n": len(df)}
    ic, p = spearmanr(df["sentiment"], df["ret_h"])
    # Hit rate: % совпадений знака
    hit = ((df["sentiment"] > 0) == (df["ret_h"] > 0)).mean()
    return {
        "source":   source_name,
        "ic":       round(ic, 4),
        "p_value":  round(p, 4),
        "hit_rate": round(hit, 4),
        "n":        len(df),
    }


# ──────────────────────────────────────────────────────────────
# 4. Lead-lag анализ
# ──────────────────────────────────────────────────────────────

def compute_lead_lag(tg_df: pl.DataFrame, news_df: pl.DataFrame, window_hours: int = 24) -> dict:
    """
    Для каждой пары (тикер, знак_сентимента) находим ближайшую новость TG
    и ближайшую новость СМИ в окне window_hours.
    Вычисляем лаг = t_сми - t_tg (в минутах).
    H0: медиана лага = 0.
    """
    print(f"\nLead-lag анализ (окно ±{window_hours}ч)...")

    tg = tg_df.to_pandas()
    news = news_df.to_pandas()

    tg["sign"] = np.sign(tg["sentiment"])
    news["sign"] = np.sign(news["sentiment"])

    # Убираем нейтральные
    tg = tg[tg["sign"] != 0].copy()
    news = news[news["sign"] != 0].copy()

    tg["event_time"] = pd.to_datetime(tg["event_time"])
    news["event_time"] = pd.to_datetime(news["event_time"])

    window = pd.Timedelta(hours=window_hours)
    lags = []  # в минутах

    tickers = set(tg["ticker"]) & set(news["ticker"])
    for ticker in tqdm(sorted(tickers), desc="   Тикеры"):
        tg_t  = tg[tg["ticker"] == ticker].sort_values("event_time")
        news_t = news[news["ticker"] == ticker].sort_values("event_time")

        if len(tg_t) == 0 or len(news_t) == 0:
            continue

        # Для каждого события TG ищем ближайшее событие СМИ того же знака
        for _, tg_row in tg_t.iterrows():
            t0 = tg_row["event_time"]
            sign = tg_row["sign"]

            candidates = news_t[
                (news_t["sign"] == sign) &
                (news_t["event_time"] >= t0 - window) &
                (news_t["event_time"] <= t0 + window)
            ]
            if len(candidates) == 0:
                continue

            # Ближайшая по времени новость СМИ
            closest = candidates.iloc[(candidates["event_time"] - t0).abs().argsort().iloc[0]]
            lag_min = (closest["event_time"] - t0).total_seconds() / 60
            lags.append(lag_min)

    if len(lags) < 30:
        return {"median_lag_min": np.nan, "mean_lag_min": np.nan, "p_value": np.nan, "n": len(lags)}

    lags = np.array(lags)
    _, p = ttest_1samp(lags, 0)
    return {
        "median_lag_min": round(float(np.median(lags)), 1),
        "mean_lag_min":   round(float(np.mean(lags)), 1),
        "pct_tg_first":   round(float((lags > 0).mean() * 100), 1),  # % случаев TG раньше
        "p_value":        round(p, 4),
        "n":              len(lags),
    }


# ──────────────────────────────────────────────────────────────
# 5. Horse-race регрессия
# ──────────────────────────────────────────────────────────────

def horse_race_regression(ev_tg: pl.DataFrame, ev_news: pl.DataFrame, horizon: str):
    """
    Объединяем TG и СМИ по (ticker, ближайшее время) и строим регрессию:
    ret_h ~ sent_tg + sent_news

    Если β_tg > β_news — TG информативнее.
    """
    tg = ev_tg.select(["ticker", "event_time", "sentiment", "ret_h"]) \
              .rename({"sentiment": "sent_tg", "event_time": "t_tg"}) \
              .to_pandas()
    news = ev_news.select(["ticker", "event_time", "sentiment", "ret_h"]) \
                  .rename({"sentiment": "sent_news", "event_time": "t_news"}) \
                  .to_pandas()

    # Для каждого события TG ищем ближайшее событие СМИ того же тикера в ±4ч
    tg["t_tg"] = pd.to_datetime(tg["t_tg"])
    news["t_news"] = pd.to_datetime(news["t_news"])

    window = pd.Timedelta(hours=4)
    merged = []

    tickers = set(tg["ticker"]) & set(news["ticker"])
    for ticker in tickers:
        tg_t = tg[tg["ticker"] == ticker]
        news_t = news[news["ticker"] == ticker]
        for _, row in tg_t.iterrows():
            t0 = row["t_tg"]
            candidates = news_t[
                (news_t["t_news"] >= t0 - window) &
                (news_t["t_news"] <= t0 + window)
            ]
            if len(candidates) == 0:
                continue
            closest = candidates.iloc[(candidates["t_news"] - t0).abs().argsort().iloc[0]]
            merged.append({
                "ticker":    ticker,
                "sent_tg":   row["sent_tg"],
                "sent_news": closest["sent_news"],
                "ret_h":     row["ret_h"],
            })

    if len(merged) < 50:
        print(f"   Недостаточно совпадающих событий ({len(merged)}) для регрессии")
        return None

    df = pd.DataFrame(merged).dropna()
    print(f"   Совпадающих событий для регрессии: {len(df):,}")

    # Стандартизируем
    for col in ["sent_tg", "sent_news"]:
        std = df[col].std()
        if std > 0:
            df[col] = df[col] / std

    model = smf.ols("ret_h ~ sent_tg + sent_news", data=df).fit(cov_type="HC3")
    return model


# ──────────────────────────────────────────────────────────────
# 6. Вывод результатов
# ──────────────────────────────────────────────────────────────

class Tee:
    """Пишет одновременно в консоль и файл."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    results_file = open(RESULTS_TXT, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, results_file)

    print("=" * 80)
    print("ГИПОТЕЗА 1: TELEGRAM ОПЕРЕЖАЕТ СМИ ПО ВЛИЯНИЮ НА ДОХОДНОСТИ")
    print(f"   Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ── Загрузка ─────────────────────────────────────────────
    tg_df   = load_tg_sentiment()
    news_df = load_news_sentiment()

    all_tickers = list(set(tg_df["ticker"].to_list()) | set(news_df["ticker"].to_list()))
    prices = load_prices(all_tickers)

    # ── Lead-lag (не зависит от горизонта) ───────────────────
    print("\n" + "=" * 80)
    print("ТЕСТ 1: LEAD-LAG — КТО ПУБЛИКУЕТ РАНЬШЕ?")
    print("=" * 80)

    ll = compute_lead_lag(tg_df, news_df, window_hours=24)
    print(f"\n  Совпадающих пар (TG ↔ СМИ):   {ll['n']:,}")
    print(f"  Медианный лаг (t_сми - t_tg): {ll['median_lag_min']:+.1f} мин")
    print(f"  Средний лаг:                  {ll['mean_lag_min']:+.1f} мин")
    print(f"  TG публикует РАНЬШЕ СМИ:      {ll['pct_tg_first']:.1f}% случаев")
    print(f"  t-test p-value (H0: лаг=0):   {ll['p_value']:.4f}")
    if ll["median_lag_min"] > 0 and ll["p_value"] < 0.05:
        print("\n  TG публикует позитивно/негативно РАНЬШЕ СМИ (статистически значимо)")
    elif ll["p_value"] >= 0.05:
        print("\n  Разница во времени публикации не значима на 5% уровне")
    else:
        print("\n  СМИ публикует раньше TG")

    # ── По горизонтам ─────────────────────────────────────────
    ic_results = []

    for horizon in HORIZONS:
        print(f"\n{'=' * 80}")
        print(f"ГОРИЗОНТ: {horizon}")
        print("=" * 80)

        # Event study TG
        print(f"\n  Строю event-study TG ({horizon})...")
        ev_tg = build_event_study(tg_df, prices, horizon)
        print(f"  TG событий: {len(ev_tg):,}")

        # Event study СМИ
        print(f"  Строю event-study СМИ ({horizon})...")
        ev_news = build_event_study(news_df, prices, horizon)
        print(f"  СМИ событий: {len(ev_news):,}")

        # ── IC сравнение ──────────────────────────────────────
        print(f"\n  ── ТЕСТ 2: INFORMATION COEFFICIENT (горизонт {horizon}) ──")
        ic_tg   = compute_ic(ev_tg,   "Telegram")
        ic_news = compute_ic(ev_news, "СМИ")

        for r in [ic_tg, ic_news]:
            print(f"  {r['source']:10} | IC={r['ic']:+.4f} | p={r['p_value']:.4f} "
                  f"| hit={r.get('hit_rate', 0):.3f} | n={r['n']:,}")

        ic_results.append({
            "horizon":       horizon,
            "ic_tg":         ic_tg["ic"],
            "p_tg":          ic_tg["p_value"],
            "hit_tg":        ic_tg.get("hit_rate", np.nan),
            "n_tg":          ic_tg["n"],
            "ic_news":       ic_news["ic"],
            "p_news":        ic_news["p_value"],
            "hit_news":      ic_news.get("hit_rate", np.nan),
            "n_news":        ic_news["n"],
        })

        tg_better = (not np.isnan(ic_tg["ic"])) and \
                    (not np.isnan(ic_news["ic"])) and \
                    (abs(ic_tg["ic"]) > abs(ic_news["ic"]))
        tg_sig    = ic_tg["p_value"] < 0.05

        if tg_better and tg_sig:
            print(f"\n  TG: |IC| больше СМИ и значим (горизонт {horizon})")
        elif tg_better:
            print(f"\n  TG: |IC| больше СМИ, но не значим (p={ic_tg['p_value']:.3f})")
        else:
            print(f"\n  СМИ: |IC| не меньше TG на горизонте {horizon}")

        # ── Horse-race регрессия ──────────────────────────────
        print(f"\n  ── ТЕСТ 3: HORSE-RACE РЕГРЕССИЯ (горизонт {horizon}) ──")
        model = horse_race_regression(ev_tg, ev_news, horizon)
        if model is not None:
            print(model.summary().tables[1])
            b_tg   = model.params.get("sent_tg", np.nan)
            b_news = model.params.get("sent_news", np.nan)
            p_tg   = model.pvalues.get("sent_tg", np.nan)
            p_news = model.pvalues.get("sent_news", np.nan)
            print(f"\n  β_tg={b_tg:+.4f} (p={p_tg:.4f})  |  β_news={b_news:+.4f} (p={p_news:.4f})")
            if abs(b_tg) > abs(b_news) and p_tg < 0.05:
                print(f"  TG-сентимент значимо сильнее СМИ в horse-race на горизонте {horizon}")
            elif p_news < 0.05 and p_tg >= 0.05:
                print(f"  Только СМИ значимы на горизонте {horizon}")
            else:
                print(f"  Ни один источник не значим на горизонте {horizon}")

    # ── Итоговая таблица IC ───────────────────────────────────
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА IC ПО ГОРИЗОНТАМ")
    print("=" * 80)
    print(f"{'Горизонт':>10} | {'IC Telegram':>12} | {'p TG':>8} | {'IC СМИ':>10} | {'p СМИ':>8} | {'TG лучше':>10}")
    print("-" * 75)
    for r in ic_results:
        tg_better = abs(r["ic_tg"]) > abs(r["ic_news"]) if not (np.isnan(r["ic_tg"]) or np.isnan(r["ic_news"])) else False
        print(f"{r['horizon']:>10} | {r['ic_tg']:>+12.4f} | {r['p_tg']:>8.4f} | "
              f"{r['ic_news']:>+10.4f} | {r['p_news']:>8.4f} | {'' if tg_better else '':>10}")

    # ── Финальный вывод ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("ВЫВОД ПО ГИПОТЕЗЕ 1")
    print("=" * 80)

    tg_wins_ic    = sum(1 for r in ic_results if not np.isnan(r["ic_tg"]) and not np.isnan(r["ic_news"]) and abs(r["ic_tg"]) > abs(r["ic_news"]))
    tg_wins_sig   = sum(1 for r in ic_results if r["p_tg"] < 0.05)
    lag_confirmed = ll["median_lag_min"] > 0 and ll["p_value"] < 0.05

    print(f"\n  TG опережает по |IC|:   {tg_wins_ic}/{len(HORIZONS)} горизонтов")
    print(f"  TG IC значим (p<0.05):  {tg_wins_sig}/{len(HORIZONS)} горизонтов")
    print(f"  TG публикует раньше:    {'ДА' if lag_confirmed else 'НЕТ'} "
          f"(лаг {ll['median_lag_min']:+.1f} мин, p={ll['p_value']:.4f})")

    if tg_wins_ic >= 3 and lag_confirmed:
        print("\n  ГИПОТЕЗА 1 ПОДТВЕРЖДАЕТСЯ: Telegram опережает СМИ")
    elif tg_wins_ic >= 2 or lag_confirmed:
        print("\n  ГИПОТЕЗА 1 ЧАСТИЧНО ПОДТВЕРЖДАЕТСЯ")
    else:
        print("\n  ГИПОТЕЗА 1 НЕ ПОДТВЕРЖДАЕТСЯ")

    print(f"\nРезультаты сохранены в: {RESULTS_TXT}")
    print("=" * 80)

    sys.stdout = original_stdout
    results_file.close()
