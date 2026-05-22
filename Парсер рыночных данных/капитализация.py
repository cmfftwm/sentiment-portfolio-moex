import requests
import json
import sqlite3
from datetime import datetime, timedelta, date
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Загружаем config.json
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

# Путь к БД капитализации (как в gipotiz.py)
DB_PATH = "капитализация.db"

# Блокировка для потокобезопасной работы с БД
db_lock = Lock()


# --- HELPERS ---
def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None


def get_tickers_from_config():
    """Получает тикеры из config.json"""
    tickers = []
    for ticker in CONFIG.get('secids', []):
        tickers.append({'ticker': ticker.upper(), 'name': ticker.upper()})
    return tickers


def ensure_ticker_table(ticker):
    """Создает таблицу для тикера, если она не существует (потокобезопасно)"""
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            safe_ticker = ticker.replace('.', '_').replace('-', '_').replace('+', '_')
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS "{safe_ticker}" (
                    date TEXT NOT NULL,
                    capitalization REAL,
                    PRIMARY KEY (date)
                )
            """)
            conn.commit()
        finally:
            conn.close()


def insert_capitalization(ticker, date_str, capitalization):
    """Вставляет или обновляет капитализацию для тикера (потокобезопасно)"""
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            safe_ticker = ticker.replace('.', '_').replace('-', '_').replace('+', '_')
            cursor.execute(f"""
                INSERT INTO "{safe_ticker}" (date, capitalization)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    capitalization = excluded.capitalization
            """, (date_str, capitalization))
            conn.commit()
        finally:
            conn.close()


def check_date_exists(ticker, date_str):
    """Проверяет, существует ли запись для тикера и даты (потокобезопасно)"""
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            safe_ticker = ticker.replace('.', '_').replace('-', '_').replace('+', '_')
            cursor.execute(f'SELECT COUNT(*) FROM "{safe_ticker}" WHERE date = ?', (date_str,))
            exists = cursor.fetchone()[0] > 0
            return exists
        finally:
            conn.close()


def fetch_capitalization_on_date(ticker, dt):
    """Получает капитализацию для тикера на указанную дату"""
    if isinstance(dt, datetime):
        date_str = dt.strftime('%Y-%m-%d')
        dt_date = dt.date()
    elif isinstance(dt, date):
        date_str = dt.strftime('%Y-%m-%d')
        dt_date = dt
    else:
        date_str = str(dt)
        dt_date = dt
    
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}.json?date={date_str}"
    try:
        resp = requests.get(url)
        data = resp.json()
        board = data.get('marketdata', {}).get('data', [])
        columns = data.get('marketdata', {}).get('columns', [])
        if not board or not columns:
            return None
        col_index = {col: i for i, col in enumerate(columns)}
        if 'ISSUECAPITALIZATION' in col_index:
            return safe_float(board[0][col_index['ISSUECAPITALIZATION']])
        return None
    except Exception as e:
        print(f"Ошибка при получении капитализации для {ticker} на {date_str}: {e}")
        time.sleep(CONFIG.get('sleep_base_sec', 1.5))
        return None


def process_ticker(ticker_data, start_date, end_date, sleep_sec, ticker_index, total_tickers):
    """Обрабатывает один тикер (для параллельного выполнения)"""
    ticker = ticker_data['ticker']
    name = ticker_data.get('name', ticker)
    
    print(f"[{ticker_index}/{total_tickers}] Сбор капитализации для {ticker} {name}")
    
    # Создаем таблицу для тикера
    ensure_ticker_table(ticker)
    
    current_date = start_date
    days_processed = 0
    days_skipped = 0
    
    while current_date <= end_date:
        # Форматируем дату как в БД (с временем)
        date_str = f"{current_date} 18:00:00.000000"
        
        # Проверяем, есть ли уже данные за эту дату
        exists = check_date_exists(ticker, date_str)
        
        if not exists:
            cap = fetch_capitalization_on_date(ticker, current_date)
            if cap is not None:
                insert_capitalization(ticker, date_str, cap)
                days_processed += 1
            else:
                days_skipped += 1
        else:
            days_skipped += 1
        
        current_date += timedelta(days=1)
        time.sleep(sleep_sec)  # Задержка из config
    
    result = {
        'ticker': ticker,
        'name': name,
        'days_processed': days_processed,
        'days_skipped': days_skipped
    }
    print(f"✓ [{ticker_index}/{total_tickers}] {ticker}: обработано {days_processed} дней (пропущено: {days_skipped})")
    return result


def main():
    # Получаем тикеры из config.json
    tickers = get_tickers_from_config()
    print(f"Тикеров из config.json: {len(tickers)}")
    
    # Получаем даты из config.json
    start_date_str = CONFIG.get('start_date', '2010-01-01')
    end_date_str = CONFIG.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    print(f"Период сбора: {start_date} - {end_date}")
    
    # Параметры из config
    sleep_sec = CONFIG.get('sleep_base_sec', 0.5)
    concurrency = CONFIG.get('concurrency', 4)
    
    print(f"Параллельных потоков: {concurrency}")
    print(f"Задержка между запросами: {sleep_sec} сек\n")
    
    # Параллельная обработка тикеров
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for idx, ticker_data in enumerate(tickers):
            future = executor.submit(
                process_ticker,
                ticker_data,
                start_date,
                end_date,
                sleep_sec,
                idx + 1,
                len(tickers)
            )
            futures.append(future)
        
        # Собираем результаты
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Ошибка при обработке тикера: {e}")
    
    # Итоговая статистика
    total_processed = sum(r['days_processed'] for r in results)
    total_skipped = sum(r['days_skipped'] for r in results)
    
    print("\n" + "=" * 60)
    print("✅ Сбор капитализации завершен")
    print(f"📊 Итого обработано дней: {total_processed:,}")
    print(f"📊 Итого пропущено дней: {total_skipped:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()


