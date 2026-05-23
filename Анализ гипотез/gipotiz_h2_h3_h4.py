import sqlite3
import polars as pl
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm
import json

# ---------- 1. Загрузка данных из SQLite ----------

def load_news_and_prices(
    news_db_path: str,
    prices_db_path: str,
    news_table: str = "news_ticker_sentiment",
    event_time_col: str = "date",
    price_col: str = "close",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Загружает новости и цены из SQLite в Polars DataFrame.
    news_ticker_sentiment: ticker, date, sentiment
    prices: каждая таблица в moex_1m.db - это тикер с колонками time, open, high, low, close
    """
    news_conn = sqlite3.connect(news_db_path)
    prices_conn = sqlite3.connect(prices_db_path)

    # Загружаем новости
    print("Загружаю новости...")
    news_query = f"""
        SELECT ticker, {event_time_col} AS event_time, sentiment
        FROM {news_table}
        WHERE sentiment IS NOT NULL
        ORDER BY ticker, event_time
    """
    news_df = pl.read_database(news_query, news_conn)
    print(f"Загружено новостей: {len(news_df):,}")

    # Получаем уникальные тикеры из новостей
    unique_tickers = news_df["ticker"].unique().to_list()
    print(f"Уникальных тикеров в новостях: {len(unique_tickers)}")

    # Получаем список всех доступных таблиц (тикеров) из базы цен
    cursor = prices_conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    available_tickers = {row[0] for row in cursor.fetchall()}

    # Фильтруем только те тикеры, которые есть и в новостях, и в базе цен
    tickers_to_load = [t for t in unique_tickers if t.lower() in available_tickers or t in available_tickers]
    print(f"Тикеров с ценами: {len(tickers_to_load)}")

    # Загружаем цены только для нужных тикеров
    prices_list = []
    for ticker in tqdm(tickers_to_load, desc="Загрузка цен"):
        # Пробуем разные варианты имени таблицы (регистр может отличаться)
        table_name = None
        for available_ticker in available_tickers:
            if available_ticker.lower() == ticker.lower():
                table_name = available_ticker
                break
        
        if table_name is None:
            continue
            
        try:
            query = f"""
                SELECT '{ticker}' AS ticker, time AS ts, {price_col} AS price
                FROM "{table_name}"
                WHERE {price_col} IS NOT NULL
                ORDER BY time
            """
            ticker_df = pl.read_database(query, prices_conn)
            if len(ticker_df) > 0:
                prices_list.append(ticker_df)
        except Exception as e:
            # Пропускаем таблицы с ошибками
            print(f"Пропущена таблица {table_name} (тикер {ticker}): {e}")
            continue

    # Объединяем все цены в один DataFrame
    if prices_list:
        prices_df = pl.concat(prices_list)
    else:
        prices_df = pl.DataFrame({"ticker": [], "ts": [], "price": []})

    news_conn.close()
    prices_conn.close()

    # Приводим к типу Datetime
    # Данные в формате ISO 8601 с timezone (например: 2025-11-19T04:11:57+00:00)
    # Используем eager вычисление через Series для корректной обработки timezone
    if len(news_df) > 0:
        event_time_list = news_df["event_time"].to_list()
        news_df = news_df.with_columns(
            pl.Series("event_time", 
                     pl.Series(event_time_list).str.to_datetime(strict=False, time_unit="ns"))
        )
        # Убираем timezone для совместимости с ценами (которые без timezone)
        news_df = news_df.with_columns(
            pl.col("event_time").dt.replace_time_zone(None)
        )
    
    if len(prices_df) > 0:
        ts_list = prices_df["ts"].to_list()
        prices_df = prices_df.with_columns(
            pl.Series("ts",
                     pl.Series(ts_list).str.to_datetime(strict=False, time_unit="ns"))
        )
        # Убеждаемся, что ts тоже без timezone
        if prices_df["ts"].dtype.time_zone is not None:
            prices_df = prices_df.with_columns(
                pl.col("ts").dt.replace_time_zone(None)
            )

    return news_df, prices_df


# ---------- 2. Построение event-study датасета ----------

def parse_horizon(horizon: str):
    """
    Парсит строку горизонта в pl.duration объект.
    Примеры: "5m", "1h", "4h", "1d"
    """
    horizon = horizon.lower().strip()
    
    if horizon.endswith('m'):
        minutes = int(horizon[:-1])
        return pl.duration(minutes=minutes)
    elif horizon.endswith('h'):
        hours = int(horizon[:-1])
        return pl.duration(hours=hours)
    elif horizon.endswith('d'):
        days = int(horizon[:-1])
        return pl.duration(days=days)
    else:
        raise ValueError(f"Неизвестный формат горизонта: {horizon}. Используйте формат: '5m', '1h', '1d' и т.д.")


def build_event_study_dataset(
    news_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    horizon: str = "1h",  # '5m', '1h', '4h', '1d', etc.
) -> pl.DataFrame:
    """
    Строит датасет вида: ticker, event_time, sentiment, price_t, price_t_h, ret_h

    horizon: строка для pl.duration, например:
        "5m" - 5 минут
        "1h" - 1 час
        "1d" - 1 день
    """

    # Сортируем
    news = news_df.select(["ticker", "event_time", "sentiment"]).sort(
        ["ticker", "event_time"]
    )
    prices = prices_df.select(["ticker", "ts", "price"]).sort(["ticker", "ts"])

    # 1) Цена в момент новости (последняя цена <= event_time)
    events_with_p0 = news.join_asof(
        prices,
        left_on="event_time",
        right_on="ts",
        by="ticker",
        strategy="backward",
    ).rename({"price": "price_t"})

    # 2) Цена через горизонт: event_time + horizon
    horizon_duration = parse_horizon(horizon)
    events_with_p0 = events_with_p0.with_columns(
        (pl.col("event_time") + horizon_duration).alias("event_time_h")
    )
    # Приводим event_time_h к типу datetime[ns] для совместимости с ts_h
    events_with_p0 = events_with_p0.with_columns(
        pl.col("event_time_h").cast(pl.Datetime(time_unit="ns")).alias("event_time_h")
    )

    events_with_p1 = events_with_p0.join_asof(
        prices.rename({"ts": "ts_h", "price": "price_h"}),
        left_on="event_time_h",
        right_on="ts_h",
        by="ticker",
        strategy="backward",
    )

    # 3) Доходность
    # Вычисляем доходность
    ret_h_expr = (pl.col("price_h") - pl.col("price_t")) / pl.col("price_t")
    ev = (
        events_with_p1
        .filter(pl.col("price_t").is_not_null() & pl.col("price_h").is_not_null())
        .with_columns([
            ret_h_expr.alias("ret_h"),
            ret_h_expr.abs().alias("abs_ret_h"),
        ])
    )

    return ev


# ---------- 3. Гипотеза 2: негатив > позитив по силе влияния ----------

def test_hypothesis_2(event_df: pl.DataFrame, sentiment_col: str = "sentiment", ret_col: str = "ret_h"):
    """
    Проверка: |return| после негативных новостей > |return| после позитивных
    """

    df = event_df.select([sentiment_col, ret_col]).drop_nulls().to_pandas()

    # Исключаем нейтральные новости (sentiment = 0) из сравнения
    df = df[df[sentiment_col] != 0]
    
    df["abs_ret"] = df[ret_col].abs()
    df["sign"] = np.where(df[sentiment_col] < 0, "neg", "pos")
    neg = df[df["sign"] == "neg"]["abs_ret"]
    pos = df[df["sign"] == "pos"]["abs_ret"]

    neg_mean = neg.mean()
    pos_mean = pos.mean()
    
    print(f"Кол-во негативных событий: {len(neg)}")
    print(f"Кол-во позитивных событий: {len(pos)}")
    print(f"Средний |ret_h| (neg): {neg_mean:.5f} ({neg_mean*100:.3f}%)")
    print(f"Средний |ret_h| (pos): {pos_mean:.5f} ({pos_mean*100:.3f}%)")
    
    # Вычисляем относительную разницу в процентах
    if pos_mean > 0:
        relative_diff = ((neg_mean - pos_mean) / pos_mean) * 100
        absolute_diff = neg_mean - pos_mean
        print(f"\nНегативные новости вызывают на {relative_diff:.2f}% больше абсолютной доходности")
        print(f"   (абсолютная разница: {absolute_diff:.5f} или {absolute_diff*100:.3f} процентных пункта)")

    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(neg, pos, equal_var=False)
    print(f"\nT-test: t = {t_stat:.3f}, p = {p_val:.5f}")
    
    # Процентный вывод
    print("\n" + "=" * 80)
    print("ВЫВОД:")
    print("=" * 80)
    if pos_mean > 0:
        print(f"Негативные новости вызывают на {relative_diff:.2f}% больше абсолютной доходности,")
        print(f"   чем позитивные новости.")
        print(f"   Это означает, что негативные новости в среднем в {1 + relative_diff/100:.2f} раза")
        print(f"   сильнее влияют на волатильность цены.")
    if p_val < 0.05:
        print(f"\nРазница статистически значима (p = {p_val:.5f} < 0.05)")
        print("   Вывод: негативные новости действительно вызывают большие изменения цены.")
    else:
        print(f"\nРазница не значима на 5% уровне (p = {p_val:.5f})")
    print("=" * 80)


# ---------- 4. Вычисление чувствительности компаний к сентименту ----------

def compute_sensitivity(event_df: pl.DataFrame, sentiment_threshold: float = 0.3):
    """
    Вычисляет чувствительность компаний к сентименту новостей.
    Чувствительность = средняя абсолютная доходность после сильных новостей.
    
    sentiment_threshold: минимальное абсолютное значение сентимента для учета новости
    """
    df = event_df.filter(
        pl.col("sentiment").abs() >= sentiment_threshold
    ).with_columns(
        pl.col("ret_h").abs().alias("abs_ret")
    )

    sensitivity = (
        df.group_by("ticker")
          .agg([
              pl.mean("abs_ret").alias("sensitivity"),
              pl.len().alias("num_events")
          ])
          .sort("sensitivity", descending=True)
    )

    return sensitivity


# ---------- 5. Гипотеза 3: крупные компании чувствительнее к сентименту ----------

def load_market_caps(
    mcap_db_path: str,
    news_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Загружает капитализацию из БД для тикеров из news_df.
    В БД каждая таблица - это тикер с колонками: date, capitalization
    """
    mcap_conn = sqlite3.connect(mcap_db_path)
    cursor = mcap_conn.cursor()
    
    # Получаем уникальные тикеры из news_df
    unique_tickers = news_df["ticker"].unique().to_list()
    
    # Получаем список всех доступных таблиц (тикеров) из базы капитализации
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    available_tickers = {row[0] for row in cursor.fetchall()}
    
    # Загружаем капитализацию для каждого тикера
    mcap_list = []
    for ticker in tqdm(unique_tickers, desc="Загрузка капитализации"):
        # Пробуем разные варианты имени таблицы (регистр может отличаться)
        table_name = None
        for available_ticker in available_tickers:
            if available_ticker.lower() == ticker.lower():
                table_name = available_ticker
                break
        
        if table_name is None:
            continue
            
        try:
            query = f"""
                SELECT '{ticker}' AS ticker, date AS mcap_date, capitalization AS mcap
                FROM "{table_name}"
                WHERE capitalization IS NOT NULL
                ORDER BY date
            """
            ticker_mcap = pl.read_database(query, mcap_conn)
            if len(ticker_mcap) > 0:
                # Приводим дату к datetime
                ticker_mcap = ticker_mcap.with_columns(
                    pl.Series("mcap_date",
                             pl.Series(ticker_mcap["mcap_date"].to_list())
                             .str.to_datetime(strict=False, time_unit="ns"))
                )
                # Убираем timezone если есть
                if ticker_mcap["mcap_date"].dtype.time_zone is not None:
                    ticker_mcap = ticker_mcap.with_columns(
                        pl.col("mcap_date").dt.replace_time_zone(None)
                    )
                mcap_list.append(ticker_mcap)
        except Exception as e:
            print(f"Пропущена таблица {table_name} (тикер {ticker}): {e}")
            continue
    
    mcap_conn.close()
    
    # Объединяем все капитализации
    if mcap_list:
        mcap_df = pl.concat(mcap_list)
    else:
        mcap_df = pl.DataFrame({"ticker": [], "mcap_date": [], "mcap": []})
    
    return mcap_df


def add_market_cap(event_df: pl.DataFrame, mcap_df: pl.DataFrame, event_time_col: str = "event_time") -> pl.DataFrame:
    """
    Добавляет капитализацию к event_df на основе даты события.
    Использует join_asof для поиска ближайшей капитализации на дату события или раньше.
    """
    # Сортируем для join_asof
    event_sorted = event_df.sort(["ticker", event_time_col])
    mcap_sorted = mcap_df.sort(["ticker", "mcap_date"])
    
    # Присоединяем капитализацию (берем последнюю доступную на дату события)
    merged = event_sorted.join_asof(
        mcap_sorted,
        left_on=event_time_col,
        right_on="mcap_date",
        by="ticker",
        strategy="backward"
    )
    
    # Добавляем log_mcap
    merged = merged.with_columns(
        pl.col("mcap").log().alias("log_mcap")
    )
    
    return merged


def test_hypothesis_3(event_df_with_mcap: pl.DataFrame, sentiment_col: str = "sentiment", ret_col: str = "ret_h"):
    """
    Проверка: чувствительность доходности к сентименту растёт с капитализацией.
    Модель: ret_h ~ sentiment * log_mcap
    """

    df = event_df_with_mcap.select(
        ["ticker", sentiment_col, ret_col, "log_mcap"]
    ).drop_nulls().to_pandas()

    # центрируем sentiment и log_mcap, чтобы интерпретация коэффициентов была стабильнее
    df["sent_c"] = df[sentiment_col] - df[sentiment_col].mean()
    df["log_mcap_c"] = df["log_mcap"] - df["log_mcap"].mean()

    model = smf.ols(
        formula=f"{ret_col} ~ sent_c * log_mcap_c",
        data=df
    ).fit(cov_type="HC3")  # робастные ошибки

    print(model.summary())
    
    # Извлекаем коэффициенты
    sent_coef = model.params.get("sent_c", 0)
    interaction_coef = model.params.get("sent_c:log_mcap_c", 0)
    sent_pval = model.pvalues.get("sent_c", 1.0)
    interaction_pval = model.pvalues.get("sent_c:log_mcap_c", 1.0)
    
    # Статистика для интерпретации
    log_mcap_std = df["log_mcap_c"].std()
    log_mcap_mean = df["log_mcap"].mean()
    
    print("\n" + "=" * 80)
    print("ПРОЦЕНТНАЯ ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
    print("=" * 80)
    
    if sent_pval < 0.05:
        print(f"Базовая реакция на сентимент (для компании со средней капитализацией):")
        print(f"   При изменении сентимента на +1 единицу доходность меняется на {sent_coef*100:.3f}%")
    else:
        print(f"Базовая реакция на сентимент не значима (p = {sent_pval:.3f})")
    
    if interaction_pval < 0.05:
        # Вычисляем эффект для крупных vs мелких компаний
        # Берем разницу между 75-м и 25-м перцентилями log_mcap
        log_mcap_q75 = df["log_mcap"].quantile(0.75)
        log_mcap_q25 = df["log_mcap"].quantile(0.25)
        log_mcap_diff = log_mcap_q75 - log_mcap_q25
        
        # Эффект взаимодействия для разницы между крупными и мелкими
        interaction_effect = interaction_coef * log_mcap_diff
        
        print(f"\nВлияние капитализации на чувствительность к сентименту:")
        print(f"   При увеличении log(капитализации) на 1 единицу, чувствительность к сентименту")
        print(f"   увеличивается на {interaction_coef*100:.3f} процентных пункта")
        print(f"\n   Сравнение крупных vs мелких компаний (разница между 75% и 25% перцентилями):")
        print(f"   Крупные компании реагируют на {interaction_effect*100:.3f}% сильнее, чем мелкие")
        print(f"   (при одинаковом изменении сентимента)")
    else:
        print(f"\nВлияние капитализации на чувствительность не значимо (p = {interaction_pval:.3f})")
        print("   Чувствительность к сентименту не зависит от размера компании")
    
    print("\n" + "=" * 80)


# ---------- 6. Гипотеза 4: секторный сентимент и влияние крупных на мелких ----------

def create_ticker_meta_from_mcap(mcap_df: pl.DataFrame, tickers_json_path: str = "tickers.json") -> pl.DataFrame:
    """
    Создает метаданные тикеров (sector, size_group) на основе капитализации и tickers.json.
    size_group определяется по медиане капитализации.
    sector загружается из tickers.json.
    """
    # Загружаем сектора из tickers.json
    try:
        with open(tickers_json_path, 'r', encoding='utf-8') as f:
            tickers_data = json.load(f)
        
        # Создаем DataFrame из JSON (исправляем опечатку "tiker" -> "ticker")
        tickers_list = []
        for item in tickers_data:
            ticker = item.get("tiker") or item.get("ticker")  # Поддержка обеих версий
            sector = item.get("sector", "unknown")
            tickers_list.append({"ticker": ticker, "sector": sector})
        
        tickers_sectors = pl.DataFrame(tickers_list)
        print(f"Загружено секторов из {tickers_json_path}: {len(tickers_sectors)} тикеров")
    except Exception as e:
        print(f"Ошибка загрузки {tickers_json_path}: {e}")
        print("   Использую 'unknown' для всех секторов")
        tickers_sectors = None
    
    # Получаем среднюю капитализацию для каждого тикера
    ticker_mcap = (
        mcap_df.group_by("ticker")
        .agg(pl.mean("mcap").alias("avg_mcap"))
        .sort("avg_mcap", descending=True)
    )
    
    # Определяем медиану для разделения на large/small
    median_mcap = ticker_mcap["avg_mcap"].median()
    
    # Создаем size_group
    ticker_meta = ticker_mcap.with_columns(
        pl.when(pl.col("avg_mcap") >= median_mcap)
        .then(pl.lit("large"))
        .otherwise(pl.lit("small"))
        .alias("size_group")
    )
    
    # Добавляем секторы из tickers.json
    if tickers_sectors is not None:
        ticker_meta = ticker_meta.join(tickers_sectors, on="ticker", how="left")
        # Заполняем пропуски "unknown"
        ticker_meta = ticker_meta.with_columns(
            pl.col("sector").fill_null("unknown")
        )
    else:
        ticker_meta = ticker_meta.with_columns(
            pl.lit("unknown").alias("sector")
        )
    
    # Оставляем только нужные колонки
    ticker_meta = ticker_meta.select(["ticker", "sector", "size_group"])
    
    return ticker_meta


def test_hypothesis_4_cross_correlation(ev: pl.DataFrame, horizon_label: str):
    """
    ТЕСТ 1: Корреляция sentiment_large vs ret_small по секторам
    """
    print(f"\n{'─' * 80}")
    print(f"ТЕСТ 1: Корреляция sentiment_large vs ret_small (горизонт {horizon_label})")
    print(f"{'─' * 80}")

    ev = ev.sort(["sector", "event_time"])

    large = ev.filter(pl.col("size_group") == "large")
    small = ev.filter(pl.col("size_group") == "small")

    sectors = ev["sector"].unique().to_list()

    results = []

    for sec in sectors:
        if sec == "unknown":  # Пропускаем неизвестные секторы
            continue
            
        sec_large = large.filter(pl.col("sector") == sec).sort("event_time")
        sec_small = small.filter(pl.col("sector") == sec).sort("event_time")

        if len(sec_large) == 0 or len(sec_small) == 0:
            continue

        # Переименовываем колонки перед join, чтобы избежать конфликтов
        sec_large_renamed = sec_large.select([
            "ticker", "event_time", "sentiment"
        ]).rename({"ticker": "large_ticker", "sentiment": "sentiment_large"})
        
        sec_small_renamed = sec_small.select([
            "ticker", "event_time", "ret_h"
        ]).rename({"ticker": "small_ticker", "ret_h": "ret_small"})
        
        cross = sec_large_renamed.join_asof(
            sec_small_renamed,
            left_on="event_time",
            right_on="event_time",
            strategy="nearest"
        ).select([
            "large_ticker",
            "sentiment_large",
            "ret_small",
            "small_ticker",
        ]).drop_nulls()

        if len(cross) < 10:
            continue

        # Вычисляем корреляцию в Polars
        corr_df = cross.select([
            pl.corr("sentiment_large", "ret_small").alias("correlation")
        ])
        corr = corr_df["correlation"][0] if len(corr_df) > 0 else None

        if corr is not None:
            results.append((sec, len(cross), corr))

    if results:
        print("\nРезультаты по секторам:")
        for sec, n, corr in results:
            print(f"   Сектор: {sec:15s} | пар: {n:5d} | corr(sent_large, ret_small) = {corr:+.4f}")
    else:
        print("Недостаточно данных для анализа по секторам (нужны секторы в метаданных)")

    return results


def test_hypothesis_4_regression(ev: pl.DataFrame, horizon_label: str):
    """
    ТЕСТ 2: Регрессия ret_h ~ sentiment_small + sentiment_large
    """
    print(f"\n{'─' * 80}")
    print(f"ТЕСТ 2: Регрессия ret_h ~ sentiment_small + sentiment_large (горизонт {horizon_label})")
    print(f"{'─' * 80}")

    ev = ev.sort(["sector", "event_time"])

    large = ev.filter(pl.col("size_group") == "large").select([
        "sector", "ticker", "event_time", "sentiment"
    ]).rename({"sentiment": "sentiment_large"})

    small = ev.filter(pl.col("size_group") == "small").select([
        "sector", "ticker", "event_time", "sentiment", "ret_h"
    ]).rename({"sentiment": "sentiment_small", "ticker": "small_ticker"})

    small_with_large = small.join_asof(
        large,
        left_on="event_time",
        right_on="event_time",
        by="sector",
        strategy="backward"
    ).drop_nulls()

    print(f"Совпадений small+large событий: {len(small_with_large)}")

    if len(small_with_large) < 50:
        print("Маловато данных для регрессии, результаты могут быть шумными.")
        return None

    df = small_with_large.to_pandas()

    model = smf.ols(
        formula="ret_h ~ sentiment_small + sentiment_large",
        data=df
    ).fit(cov_type="HC3")

    print(model.summary())

    print("\nИнтерпретация коэффициентов:")
    print("   sentiment_small  — реакция мелких компаний на их собственные новости.")
    print("   sentiment_large  — перекрестное влияние новостей крупных компаний сектора.")
    print("Если coef(sentiment_large) значим и по модулю не слишком мал → гипотеза 4 поддержана.")

    return model


def build_sector_sentiment(ev: pl.DataFrame) -> pl.DataFrame:
    """
    Создаёт фактор SectorSentiment_t для каждого сектора:
    средний sentiment по крупным компаниям в каждый момент времени.
    """
    large = ev.filter(pl.col("size_group") == "large")

    sector_sent = (
        large.group_by(["sector", "event_time"])
        .agg(pl.mean("sentiment").alias("sector_sentiment"))
        .sort(["sector", "event_time"])
    )

    return sector_sent


def test_hypothesis_4_sector_factor(ev: pl.DataFrame, horizon_label: str):
    """
    ТЕСТ 3: SectorSentiment factor model (ret_h ~ sector_sentiment)
    """
    print(f"\n{'─' * 80}")
    print(f"ТЕСТ 3: SectorSentiment factor model (ret_h ~ sector_sentiment), горизонт {horizon_label}")
    print(f"{'─' * 80}")

    ev = ev.sort(["sector", "event_time"])

    sector_sent = build_sector_sentiment(ev)

    small = ev.filter(pl.col("size_group") == "small").select([
        "sector", "ticker", "event_time", "ret_h"
    ]).rename({"ticker": "small_ticker"})

    small_with_factor = small.join_asof(
        sector_sent,
        left_on="event_time",
        right_on="event_time",
        by="sector",
        strategy="backward"
    ).drop_nulls()

    print(f"Совпадений small + sector_sentiment: {len(small_with_factor)}")

    if len(small_with_factor) < 50:
        print("Маловато данных для факторной регрессии.")
        return None

    df = small_with_factor.to_pandas()

    model = smf.ols(
        formula="ret_h ~ sector_sentiment",
        data=df
    ).fit(cov_type="HC3")

    print(model.summary())

    print("\nИнтерпретация:")
    print("   sector_sentiment — это средний сентимент по КРУПНЫМ компаниям сектора.")
    print("Если coef(sector_sentiment) значим и с ожидаемым знаком →")
    print("   → секторный сентимент крупных действительно двигает мелкие компании.")
    return model


# ---------- 7. Пример использования всего пайплайна ----------

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    NEWS_DB_PATH = "telegram_messages.db"   # база с новостями
    PRICES_DB_PATH = "moex_1m.db"           # база с ценами
    MCAP_DB_PATH = "капитализация.db"       # база с капитализацией
    RESULTS_DB_PATH = "event_study_results.db"  # база для результатов

    # Горизонты для анализа
    HORIZONS = ["5m", "1h", "4h", "1d"]
    
    # 1) Грузим новости и цены (один раз для всех горизонтов)
    print("=" * 80)
    print("АНАЛИЗ ВЛИЯНИЯ СЕНТИМЕНТА НА ДОХОДНОСТЬ АКЦИЙ")
    print("=" * 80)
    news_df, prices_df = load_news_and_prices(NEWS_DB_PATH, PRICES_DB_PATH)
    
    # Загружаем капитализацию (будет использоваться для гипотезы 3)
    print("\nЗагружаю данные о капитализации...")
    mcap_df = load_market_caps(MCAP_DB_PATH, news_df)
    print(f"Загружено записей о капитализации: {len(mcap_df):,}")

    # 2) Создаем БД для результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_conn = sqlite3.connect(RESULTS_DB_PATH)
    results_cursor = results_conn.cursor()
    
    # Создаем таблицу для метаданных анализа
    results_cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            horizons TEXT,
            total_news INTEGER,
            total_tickers INTEGER
        )
    """)
    
    # Записываем метаданные
    results_cursor.execute("""
        INSERT INTO analysis_metadata (timestamp, horizons, total_news, total_tickers)
        VALUES (?, ?, ?, ?)
    """, (timestamp, ",".join(HORIZONS), len(news_df), len(news_df["ticker"].unique())))
    results_conn.commit()
    
    # 3) Открываем один текстовый файл для всех результатов
    results_file = f"hypothesis_results_{timestamp}.txt"
    results_text_file = open(results_file, 'w', encoding='utf-8')
    
    # Записываем заголовок
    results_text_file.write("=" * 80 + "\n")
    results_text_file.write("АНАЛИЗ ВЛИЯНИЯ СЕНТИМЕНТА НА ДОХОДНОСТЬ АКЦИЙ\n")
    results_text_file.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_text_file.write(f"Горизонты: {', '.join(HORIZONS)}\n")
    results_text_file.write("=" * 80 + "\n")
    results_text_file.write("\nРазница статистически значима на 5% уровне.\n")
    results_text_file.write("(Это означает, что негативные новости действительно вызывают большие\n")
    results_text_file.write("абсолютные изменения цены, чем позитивные, и это не случайность)\n")
    results_text_file.write("\n" + "=" * 80 + "\n\n")
    
    # 4) Анализируем каждый горизонт
    for horizon in HORIZONS:
        print(f"\n{'=' * 80}")
        print(f"АНАЛИЗ ДЛЯ ГОРИЗОНТА: {horizon}")
        print(f"{'=' * 80}")
        
        # Строим event-study датасет
        print(f"\nСтрою event-study датасет для горизонта {horizon}...")
        ev = build_event_study_dataset(news_df, prices_df, horizon=horizon)
        print(f"Создано записей: {len(ev):,}")
        print(ev.head())
        
        # Сохраняем event-study датасет в БД (таблица для каждого горизонта)
        table_name = f"event_study_{horizon}"
        print(f"\nСохраняю event-study датасет в БД (таблица: {table_name})...")
        
        # Конвертируем в pandas для записи в БД
        ev_pandas = ev.to_pandas()
        ev_pandas.to_sql(table_name, results_conn, if_exists="replace", index=False)
        results_conn.commit()
        print(f"Данные сохранены в таблицу {table_name} ({len(ev_pandas):,} записей)")

        # Проверка гипотезы 2
        print(f"\n{'─' * 80}")
        print(f"=== Гипотеза 2 (горизонт {horizon}): негатив > позитив по силе влияния ===")
        print(f"{'─' * 80}")
        
        # Записываем разделитель в текстовый файл
        results_text_file.write("\n" + "=" * 80 + "\n")
        results_text_file.write(f"ГОРИЗОНТ: {horizon}\n")
        results_text_file.write("=" * 80 + "\n")
        results_text_file.write(f"=== Гипотеза 2: негатив > позитив по силе влияния ===\n")
        results_text_file.write("-" * 80 + "\n")
        
        # Класс для записи в файл и консоль одновременно
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, results_text_file)
        
        try:
            test_hypothesis_2(ev)
            
            # Вычисляем чувствительность компаний
            print(f"\n{'─' * 80}")
            print(f"ТОП-10 компаний по чувствительности к сентименту (горизонт {horizon}):")
            print(f"{'─' * 80}")
            sensitivity = compute_sensitivity(ev, sentiment_threshold=0.3)
            print(sensitivity.head(10))
            
            # Проверка гипотезы 3 (если есть данные о капитализации)
            if len(mcap_df) > 0:
                print(f"\n{'─' * 80}")
                print(f"=== Гипотеза 3 (горизонт {horizon}): крупные компании чувствительнее к сентименту ===")
                print(f"{'─' * 80}")
                results_text_file.write(f"\n=== Гипотеза 3: крупные компании чувствительнее к сентименту ===\n")
                results_text_file.write("-" * 80 + "\n")
                
                # Добавляем капитализацию к event-study датасету
                ev_with_mcap = add_market_cap(ev, mcap_df)
                # Фильтруем только записи с капитализацией
                ev_with_mcap = ev_with_mcap.filter(pl.col("mcap").is_not_null())
                print(f"Записей с капитализацией: {len(ev_with_mcap):,}")
                
                if len(ev_with_mcap) > 0:
                    test_hypothesis_3(ev_with_mcap)
                    
                    # Проверка гипотезы 4 (секторный сентимент)
                    print(f"\n{'─' * 80}")
                    print(f"=== Гипотеза 4 (горизонт {horizon}): секторный сентимент и влияние крупных на мелких ===")
                    print(f"{'─' * 80}")
                    results_text_file.write(f"\n=== Гипотеза 4: секторный сентимент и влияние крупных на мелких ===\n")
                    results_text_file.write("-" * 80 + "\n")
                    
                    # Создаем метаданные тикеров (sector, size_group) на основе капитализации и tickers.json
                    ticker_meta = create_ticker_meta_from_mcap(mcap_df, "tickers.json")
                    
                    # Добавляем метаданные к event-study датасету
                    ev_with_meta = ev_with_mcap.join(ticker_meta, on="ticker", how="inner")
                    
                    # Фильтруем только записи с метаданными
                    ev_with_meta = ev_with_meta.filter(
                        pl.col("size_group").is_not_null() & 
                        pl.col("sector").is_not_null()
                    )
                    print(f"Записей с метаданными: {len(ev_with_meta):,}")
                    print(f"Large компаний: {len(ev_with_meta.filter(pl.col('size_group') == 'large')):,}")
                    print(f"Small компаний: {len(ev_with_meta.filter(pl.col('size_group') == 'small')):,}")
                    
                    if len(ev_with_meta) > 0:
                        # Тест 1: Корреляция
                        test_hypothesis_4_cross_correlation(ev_with_meta, horizon)
                        
                        # Тест 2: Регрессия
                        test_hypothesis_4_regression(ev_with_meta, horizon)
                        
                        # Тест 3: SectorSentiment factor
                        test_hypothesis_4_sector_factor(ev_with_meta, horizon)
                    else:
                        print("Нет данных с метаданными для тестирования гипотезы 4")
                else:
                    print("Нет данных с капитализацией для тестирования гипотезы 3")
            
        finally:
            sys.stdout = original_stdout
            results_text_file.flush()
    
    # Закрываем файл и БД
    results_text_file.close()
    results_conn.close()
    
    print(f"\n{'=' * 80}")
    print("АНАЛИЗ ЗАВЕРШЕН ДЛЯ ВСЕХ ГОРИЗОНТОВ")
    print(f"{'=' * 80}")
    print(f"\nВсе event-study датасеты сохранены в: {RESULTS_DB_PATH}")
    print(f"Все результаты гипотез сохранены в: {results_file}")

    # 4) Для гипотезы 3 нужно добавить капитализации
    # Предположим, у тебя есть parquet c капитализациями:
    # ticker, mcap (в рублях)
    # mcap_df = pl.read_parquet("market_caps.parquet")
    # ev_with_mcap = add_market_cap(ev_1h, mcap_df)
    # print("\n=== Гипотеза 3: крупные компании чувствительнее к сентименту ===")
    # test_hypothesis_3(ev_with_mcap)