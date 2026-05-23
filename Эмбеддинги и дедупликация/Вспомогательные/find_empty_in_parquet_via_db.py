"""
Проверка parquet файлов через БД - находит эмбеддинги для пустых сообщений
Сравнивает message_id из parquet с БД и проверяет, пустое ли сообщение
"""

import polars as pl
import numpy as np
import os
import sqlite3
from tqdm import tqdm

# Пути
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Папка находится в корне проекта, на уровень выше от скрипта
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET_BASE_FOLDER = os.path.join(PROJECT_ROOT, "Парквайт эмбенндинг")

# БД
possible_db_paths = [
    os.path.join(SCRIPT_DIR, "telegram_messages.db"),
    "telegram_messages.db",
    "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages.db",
]

db_path = None
for path in possible_db_paths:
    if os.path.exists(path):
        db_path = path
        break

if not db_path:
    print("БД не найдена")
    exit(1)

print("=" * 60)
print("ПРОВЕРКА PARQUET ФАЙЛОВ ЧЕРЕЗ БД")
print("=" * 60)
print(f"Папка: {PARQUET_BASE_FOLDER}")
print(f"БД: {db_path}")
print("=" * 60)

# Ищем все parquet файлы
parquet_files = []
if os.path.exists(PARQUET_BASE_FOLDER):
    for root, dirs, files in os.walk(PARQUET_BASE_FOLDER):
        for file in files:
            if file.startswith("embeddings_") and file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                parquet_files.append(file_path)

parquet_files = sorted(parquet_files, key=lambda x: os.path.basename(x))

if not parquet_files:
    print("Parquet файлы не найдены")
    exit(1)

print(f"\nНайдено {len(parquet_files)} файлов\n")

# Подключаемся к БД
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Результаты
all_results = []
total_empty_in_parquet = 0

for file_path in tqdm(parquet_files, desc="Проверка файлов"):
    channel_name = os.path.basename(file_path).replace("embeddings_", "").replace(".parquet", "")
    
    try:
        # Читаем parquet
        df = pl.read_parquet(file_path)
        
        if len(df) == 0:
            continue
        
        # Получаем все message_id из parquet
        parquet_ids = df["message_id"].to_list()
        
        # Проверяем в БД - какие из этих message_id имеют пустые сообщения
        # Батчами по 1000 для эффективности
        empty_ids = []
        
        for i in range(0, len(parquet_ids), 1000):
            batch = parquet_ids[i:i+1000]
            placeholders = ','.join('?' * len(batch))
            
            try:
                query = f"""
                    SELECT message_id 
                    FROM {channel_name}
                    WHERE message_id IN ({placeholders})
                    AND (message IS NULL OR message = '' OR TRIM(message) = '')
                """
                cursor.execute(query, batch)
                batch_empty = [row[0] for row in cursor.fetchall()]
                empty_ids.extend(batch_empty)
            except sqlite3.OperationalError as e:
                # Таблица может не существовать
                if "no such table" in str(e).lower():
                    print(f"   Таблица {channel_name} не найдена в БД")
                break
        
        empty_count = len(empty_ids)
        
        all_results.append({
            'channel': channel_name,
            'total': len(df),
            'empty_in_db': empty_count,
            'empty_ids': empty_ids[:10] if empty_ids else []  # Первые 10 для примера
        })
        
        total_empty_in_parquet += empty_count
        
    except Exception as e:
        print(f"Ошибка при обработке {channel_name}: {e}")
        all_results.append({
            'channel': channel_name,
            'total': 0,
            'empty_in_db': 0,
            'error': str(e)
        })

conn.close()

# Выводим результаты
print(f"\n{'=' * 80}")
print(f"{'Канал':<40} {'Всего':<10} {'Пустых в БД':<15} {'Статус':<15}")
print("=" * 80)

total_messages = 0
channels_with_empty = 0

for r in all_results:
    channel = r['channel']
    total = r['total']
    empty = r['empty_in_db']
    
    if 'error' in r:
        status = f"{r['error'][:12]}"
    elif empty == 0:
        status = "OK"
    else:
        status = f"{empty} пустых"
        channels_with_empty += 1
    
    print(f"{channel:<40} {total:>9,} {empty:>14,} {status:<15}")
    
    if empty > 0 and r.get('empty_ids'):
        print(f"   Примеры пустых message_id: {r['empty_ids']}")
    
    total_messages += total

print("=" * 80)
print(f"{'ИТОГО':<40} {total_messages:>9,} {total_empty_in_parquet:>14,}")

if channels_with_empty > 0:
    print(f"\nНайдено {channels_with_empty} файлов с эмбеддингами для пустых сообщений")
    print(f"   Всего пустых записей в parquet: {total_empty_in_parquet:,}")
    print(f"\nЗапустите clean_empty_embeddings.py для очистки")
else:
    print(f"\nВсе файлы в порядке - эмбеддингов для пустых сообщений не найдено")
