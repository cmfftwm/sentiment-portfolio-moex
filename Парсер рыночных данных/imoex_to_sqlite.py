"""
imoex_to_sqlite.py

Парсер дневных свечей индекса IMOEX (Московская биржа) в SQLite.

Источник: MOEX ISS API
  engine=stock, market=index, boardid=SNDX, secid=IMOEX

Таблица: imoex (в moex_indices.db)
  time, open, high, low, close, value, volume, boardid

Использование:
  python imoex_to_sqlite.py --from 2019-01-01 --till 2025-12-31
  python imoex_to_sqlite.py --from 2024-01-01 --till 2024-12-31 --interval 60
  python imoex_to_sqlite.py --from 2019-01-01 --till 2025-12-31 --db moex_indices.db
"""

import argparse
import datetime as dt
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import requests


# ─── Константы ───────────────────────────────────────────────────────────────

ISS_BASE  = "https://iss.moex.com/iss"
SECID     = "IMOEX"
ENGINE    = "stock"
MARKET    = "index"
BOARDID   = "SNDX"
TABLE     = "imoex"

# Интервалы MOEX ISS: 1=1мин, 10=10мин, 60=1час, 24=1день, 7=1неделя, 31=1месяц
DEFAULT_INTERVAL = 24   # дневные свечи

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB  = os.path.join(BASE_DIR, "moex_indices.db")


# ─── Логирование ─────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Инициализация БД ─────────────────────────────────────────────────────────

def ensure_db(db_path: str) -> None:
    new_db = not os.path.exists(db_path)
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=MEMORY")
        if new_db:
            con.execute("PRAGMA page_size=32768")
        con.executescript(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
              time    DATETIME NOT NULL,
              open    REAL     NOT NULL,
              high    REAL     NOT NULL,
              low     REAL     NOT NULL,
              close   REAL     NOT NULL,
              value   REAL,
              volume  REAL,
              boardid TEXT     NOT NULL,
              PRIMARY KEY (time)
            );

            CREATE INDEX IF NOT EXISTS idx_{TABLE}_time
            ON {TABLE}(time);
        """)
        con.commit()
        log(f"БД инициализирована: {db_path}")
    finally:
        con.close()


# ─── HTTP запросы с ретраями ──────────────────────────────────────────────────

def request_json(
    url: str,
    params: Dict,
    timeout: int,
    max_retries: int,
    sleep_base: float,
) -> Dict:
    attempt = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = sleep_base * (2 ** (attempt - 1))
            log(f"Предупреждение: {e}. Повтор через {sleep_s:.1f}с (попытка {attempt}/{max_retries})...")
            time.sleep(sleep_s)


# ─── Загрузка свечей за один день (с пагинацией) ──────────────────────────────

def fetch_day_candles(
    day: dt.date,
    interval: int,
    timeout: int,
    max_retries: int,
    sleep_base: float,
) -> List[Dict]:
    url = f"{ISS_BASE}/engines/{ENGINE}/markets/{MARKET}/securities/{SECID}/candles.json"
    offset = 0
    rows: List[Dict] = []

    while True:
        params = {
            "from":     day.isoformat(),
            "till":     day.isoformat(),
            "interval": interval,
            "start":    offset,
        }
        j = request_json(url, params, timeout, max_retries, sleep_base)
        block = j.get("candles", {})
        data  = block.get("data", [])
        cols  = block.get("columns", [])

        if not data:
            break

        cidx       = {c: i for i, c in enumerate(cols)}
        has_value  = "value"  in cidx
        has_volume = "volume" in cidx

        for d in data:
            rows.append({
                "time":   d[cidx["end"]],
                "open":   d[cidx["open"]],
                "high":   d[cidx["high"]],
                "low":    d[cidx["low"]],
                "close":  d[cidx["close"]],
                "value":  d[cidx["value"]]  if has_value  else None,
                "volume": d[cidx["volume"]] if has_volume else None,
            })
        offset += len(data)

    return rows


# ─── Upsert свечей в БД ───────────────────────────────────────────────────────

def upsert_candles(db_path: str, rows: List[Dict]) -> int:
    if not rows:
        return 0
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            f"""
            INSERT INTO {TABLE} (time, open, high, low, close, value, volume, boardid)
            VALUES (:time, :open, :high, :low, :close, :value, :volume, :boardid)
            ON CONFLICT(time) DO UPDATE SET
              open    = excluded.open,
              high    = excluded.high,
              low     = excluded.low,
              close   = excluded.close,
              value   = COALESCE(excluded.value,  {TABLE}.value),
              volume  = COALESCE(excluded.volume, {TABLE}.volume),
              boardid = excluded.boardid
            """,
            [{**r, "boardid": BOARDID} for r in rows],
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


# ─── Итерация по дням ─────────────────────────────────────────────────────────

def iter_days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


# ─── Основной цикл ────────────────────────────────────────────────────────────

def run(
    start_date: dt.date,
    end_date: dt.date,
    interval: int,
    db_path: str,
    timeout: int,
    max_retries: int,
    sleep_base: float,
) -> None:
    ensure_db(db_path)

    log(f"IMOEX [{ENGINE}/{MARKET}/{BOARDID}], interval={interval}")
    log(f"Период: {start_date} → {end_date}")
    log(f"БД: {db_path}")

    n_days        = (end_date - start_date).days + 1
    total_upserted = 0
    done           = 0

    for day in iter_days(start_date, end_date):
        done += 1
        try:
            rows = fetch_day_candles(day, interval, timeout, max_retries, sleep_base)
            if rows:
                added = upsert_candles(db_path, rows)
                total_upserted += added
                log(f"{day} — {added:4d} записей  [{done}/{n_days}]")
            else:
                log(f"{day} — нет данных (выходной/праздник)  [{done}/{n_days}]")
        except Exception as e:
            log(f"Ошибка на {day}: {e} — продолжаю.")

    log(f"\n=== ИТОГО: вставлено/обновлено {total_upserted} свечей ===")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка свечей IMOEX из MOEX ISS в SQLite"
    )
    parser.add_argument(
        "--from", dest="date_from", required=True,
        help="Начало периода, формат YYYY-MM-DD"
    )
    parser.add_argument(
        "--till", dest="date_till", required=True,
        help="Конец периода, формат YYYY-MM-DD"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Интервал свечей (1/10/60/24/7/31), по умолчанию {DEFAULT_INTERVAL} (дневные)"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"Путь к SQLite-файлу, по умолчанию {DEFAULT_DB}"
    )
    parser.add_argument("--timeout",     type=int,   default=30,  help="Таймаут HTTP запроса (сек)")
    parser.add_argument("--max-retries", type=int,   default=5,   help="Максимум ретраев")
    parser.add_argument("--sleep-base",  type=float, default=1.5, help="База для экспоненциальной паузы (сек)")

    args = parser.parse_args()

    run(
        start_date  = parse_date(args.date_from),
        end_date    = parse_date(args.date_till),
        interval    = args.interval,
        db_path     = args.db,
        timeout     = args.timeout,
        max_retries = args.max_retries,
        sleep_base  = args.sleep_base,
    )


if __name__ == "__main__":
    main()
