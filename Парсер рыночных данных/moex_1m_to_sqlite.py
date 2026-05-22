import argparse
import datetime as dt
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import requests


ISS_BASE = "https://iss.moex.com/iss"
DEFAULT_DB = "moex_1m.sqlite"


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # валидация
    for key in ["start_date", "end_date"]:
        if key not in cfg or not cfg[key]:
            raise ValueError(f"В config.json отсутствует обязательное поле '{key}'.")
    
    # поддержка как одного тикера, так и списка тикеров
    if "secid" in cfg and cfg["secid"]:
        if isinstance(cfg["secid"], str):
            cfg["secids"] = [cfg["secid"]]
        elif isinstance(cfg["secid"], list):
            cfg["secids"] = cfg["secid"]
        else:
            raise ValueError("Поле 'secid' должно быть строкой или списком строк.")
    elif "secids" in cfg and cfg["secids"]:
        if not isinstance(cfg["secids"], list):
            raise ValueError("Поле 'secids' должно быть списком строк.")
    else:
        raise ValueError("В config.json отсутствует обязательное поле 'secid' или 'secids'.")
    
    # нормализация
    cfg.setdefault("interval", 1)
    cfg.setdefault("db_name", DEFAULT_DB)
    cfg.setdefault("timeout_sec", 30)
    cfg.setdefault("max_retries", 5)
    cfg.setdefault("sleep_base_sec", 1.5)
    return cfg


def ensure_db(db_path: str) -> None:
    """Инициализирует базу данных с базовыми настройками."""
    new_db = not os.path.exists(db_path)
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=MEMORY")
        # page_size меняют до создания таблиц; если новая БД — применим
        if new_db:
            con.execute("PRAGMA page_size=32768")
        con.commit()
    finally:
        con.close()


def ensure_secid_table(db_path: str, secid: str) -> str:
    """Создает таблицу для конкретной акции, если она не существует."""
    # Очищаем secid от специальных символов для имени таблицы
    safe_secid = secid.replace('.', '_').replace('-', '_').replace('+', '_')
    table_name = safe_secid.lower()
    
    con = sqlite3.connect(db_path)
    try:
        # Создаем таблицу для акции (IF NOT EXISTS предотвращает ошибку при повторном создании)
        con.executescript(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              time DATETIME NOT NULL,
              open REAL NOT NULL,
              high REAL NOT NULL,
              low REAL NOT NULL,
              close REAL NOT NULL,
              value REAL,
              volume REAL,
              boardid TEXT NOT NULL,
              PRIMARY KEY (time)
            );

            CREATE INDEX IF NOT EXISTS idx_{table_name}_time
            ON {table_name}(time);
        """)
        
        con.commit()
        return table_name
    finally:
        con.close()


def request_json(url: str, params: Dict, timeout: int, max_retries: int, sleep_base: float) -> Dict:
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
            # экспоненциальная пауза
            sleep_s = sleep_base * (2 ** (attempt - 1))
            log(f"Предупреждение: {e}. Ретраим через {sleep_s:.1f} c (попытка {attempt}/{max_retries})...")
            time.sleep(sleep_s)


def autodetect_route_for_secid(
    secid: str, timeout: int, max_retries: int, sleep_base: float
) -> Tuple[str, str, str]:
    """
    Пытаемся определить engine/market/boardid для заданного SECID.
    Алгоритм:
     1) /iss/securities/{secid}.json -> boards
     2) если есть TQBR — берём как (stock/shares/TQBR)
     3) иначе берём первую доску с is_primary=1
     4) иначе — первую доступную, и пытаемся проецировать engine/market по board_group
    Для большинства ликвидных акций TQBR достаточно.
    """
    url = f"{ISS_BASE}/securities/{secid}.json"
    j = request_json(url, {}, timeout, max_retries, sleep_base)

    boards = j.get("boards", {}).get("data", [])
    cols = j.get("boards", {}).get("columns", [])
    idx = {c: i for i, c in enumerate(cols)}

    if not boards:
        # запасной вариант: считаем, что акция в TQBR
        log("Автодетект: не нашли boards. Предположим stock/shares/TQBR.")
        return "stock", "shares", "TQBR"

    # ищем TQBR (акции T+)
    for b in boards:
        if b[idx.get("boardid")] == "TQBR":
            return "stock", "shares", "TQBR"

    # иначе — первая первичная доска
    primary = [b for b in boards if b[idx.get("is_primary")] == 1]
    if primary:
        bid = primary[0][idx.get("boardid")]
        engine, market = guess_engine_market_from_board(bid, boards, idx)
        return engine, market, bid

    # иначе — первая попавшаяся
    bid = boards[0][idx.get("boardid")]
    engine, market = guess_engine_market_from_board(bid, boards, idx)
    return engine, market, bid


def guess_engine_market_from_board(bid: str, boards: List[List], idx: Dict[str, int]) -> Tuple[str, str]:
    """
    Очень простая эвристика: по board_group к линии engines/markets.
    На практике для акций TQBR → stock/shares, фьючи (RFUD/RFUD*) → futures/forts и т.п.
    Если определить не получилось — оставляем "stock"/"shares".
    """
    # пробуем найти запись по этой boardid, глянуть board_group
    group = None
    for b in boards:
        if b[idx.get("boardid")] == bid:
            group = b[idx.get("board_group_name")] or b[idx.get("board_group_id")]
            break

    if isinstance(group, str):
        g = group.lower()
        if "share" in g or "акции" in g:
            return "stock", "shares"
        if "bond" in g or "облигац" in g:
            return "stock", "bonds"
        if "futures" in g or "фор" in g or "deriv" in g:
            return "futures", "forts"
        if "currency" in g or "валют" in g:
            return "currency", "selt"
        if "index" in g or "индекс" in g:
            return "stock", "index"

    # дефолт для непонятного борда — акции
    return "stock", "shares"


def iter_days(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def fetch_day_candles(
    engine: str,
    market: str,
    secid: str,
    day: dt.date,
    interval: int,
    timeout: int,
    max_retries: int,
    sleep_base: float,
) -> Tuple[List[Dict], Optional[List[str]]]:
    """
    Возвращает список словарей с полями свечи и список колонок (для отладки).
    Пагинируем по 'start' до пустого ответа.
    """
    url = f"{ISS_BASE}/engines/{engine}/markets/{market}/securities/{secid}/candles.json"
    start = 0
    rows: List[Dict] = []
    columns = None
    while True:
        params = {
            "from": day.isoformat(),
            "till": day.isoformat(),
            "interval": interval,
            "start": start,
        }
        j = request_json(url, params, timeout, max_retries, sleep_base)
        block = j.get("candles", {})
        data = block.get("data", [])
        cols = block.get("columns", [])
        if columns is None:
            columns = cols

        if not data:
            break

        cidx = {c: i for i, c in enumerate(cols)}
        # некоторые поля могут отсутствовать (value)
        has_value = "value" in cidx
        has_volume = "volume" in cidx

        for d in data:
            rows.append(
                {
                    "time": d[cidx["end"]],
                    "open": d[cidx["open"]],
                    "high": d[cidx["high"]],
                    "low": d[cidx["low"]],
                    "close": d[cidx["close"]],
                    "value": d[cidx["value"]] if has_value else None,
                    "volume": d[cidx["volume"]] if has_volume else None,
                }
            )
        start += len(data)

    return rows, columns


def upsert_candles(
    db_path: str,
    secid: str,
    boardid: str,
    rows: List[Dict],
    table_name: str,
) -> int:
    """Вставляет данные в таблицу конкретной акции."""
    if not rows:
        return 0
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            f"""
            INSERT INTO {table_name} (
              time, open, high, low, close, value, volume, boardid
            )
            VALUES (
              :time, :open, :high, :low, :close, :value, :volume, :boardid
            )
            ON CONFLICT(time) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              value=COALESCE(excluded.value, {table_name}.value),
              volume=COALESCE(excluded.volume, {table_name}.volume),
              boardid=excluded.boardid
            """,
            [
                {
                    "time": r["time"],
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "value": r["value"],
                    "volume": r["volume"],
                    "boardid": boardid,
                }
                for r in rows
            ],
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def process_secid(secid: str, start_date: dt.date, end_date: dt.date, interval: int, 
                  db_path: str, timeout: int, max_retries: int, sleep_base: float,
                  engine: str = None, market: str = None, boardid: str = None) -> int:
    """Обрабатывает один тикер и возвращает количество вставленных записей."""
    secid = secid.upper().strip()
    
    if not (engine and market and boardid):
        log(f"Автодетект маршрута для {secid} ...")
        engine, market, boardid = autodetect_route_for_secid(
            secid, timeout, max_retries, sleep_base
        )
        log(f"→ engine={engine}, market={market}, boardid={boardid}")

    # Создаем таблицу для акции
    table_name = ensure_secid_table(db_path, secid)
    log(f"Используется таблица: {table_name}")

    log(
        f"Старт выгрузки: {secid} [{engine}/{market}/{boardid}], "
        f"{start_date} → {end_date}, interval={interval}"
    )

    total_inserted = 0
    n_days = (end_date - start_date).days + 1
    done = 0

    for day in iter_days(start_date, end_date):
        done += 1
        try:
            rows, cols = fetch_day_candles(
                engine, market, secid, day, interval, timeout, max_retries, sleep_base
            )
            if rows:
                added = upsert_candles(db_path, secid, boardid, rows, table_name)
                total_inserted += added
                log(f"{secid} {day} — {added:4d} мин.  [{done}/{n_days}]")
            else:
                # это нормально: выходной/праздник/нет торгов
                log(f"{secid} {day} — нет данных.      [{done}/{n_days}]")
        except Exception as e:
            log(f"Ошибка на {day}: {e} — продолжаю со следующим днём.")
            # можно добавить запись в отдельный лог-файл / список «проблемных» дат

    log(f"Тикер {secid} завершён. Вставлено/обновлено минут: {total_inserted}.")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="MOEX 1-minute candles to SQLite")
    parser.add_argument("--config", default="config.json", help="Путь к config.json")
    args = parser.parse_args()

    cfg = read_config(args.config)
    secids = [secid.upper().strip() for secid in cfg["secids"]]
    start_date = parse_date(cfg["start_date"])
    end_date = parse_date(cfg["end_date"])
    interval = int(cfg.get("interval", 1))
    db_path = cfg.get("db_name", DEFAULT_DB)

    timeout = int(cfg.get("timeout_sec", 30))
    max_retries = int(cfg.get("max_retries", 5))
    sleep_base = float(cfg.get("sleep_base_sec", 1.5))

    ensure_db(db_path)

    # общие настройки для всех тикеров (если указаны)
    engine = cfg.get("engine")
    market = cfg.get("market")
    boardid = cfg.get("boardid")

    log(f"Обработка {len(secids)} тикеров: {', '.join(secids)}")
    log(f"Период: {start_date} → {end_date}, interval={interval}, db={db_path}")

    grand_total = 0
    for i, secid in enumerate(secids, 1):
        log(f"\n=== Тикер {i}/{len(secids)}: {secid} ===")
        try:
            inserted = process_secid(
                secid, start_date, end_date, interval, db_path,
                timeout, max_retries, sleep_base, engine, market, boardid
            )
            grand_total += inserted
        except Exception as e:
            log(f"Критическая ошибка для тикера {secid}: {e}")
            continue

    log(f"\n=== ИТОГО ===\nОбработано тикеров: {len(secids)}\nВставлено/обновлено минут: {grand_total}")


if __name__ == "__main__":
    main()