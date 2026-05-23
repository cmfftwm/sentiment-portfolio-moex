"""
Проверка всех parquet файлов на наличие пустых эмбеддингов
"""

import polars as pl
import numpy as np
import os
import sqlite3
from tqdm import tqdm

# Путь к папке с parquet файлами
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_BASE_FOLDER = os.path.join(SCRIPT_DIR, "Парквайт эмбенндинг")

# Если основной папки нет, ищем в текущей директории
if not os.path.exists(PARQUET_BASE_FOLDER):
    PARQUET_BASE_FOLDER = os.path.join(SCRIPT_DIR, ".")
    for item in os.listdir(SCRIPT_DIR):
        item_path = os.path.join(SCRIPT_DIR, item)
        if os.path.isdir(item_path) and ("парквайт" in item.lower() or "parquet" in item.lower() or "embedding" in item.lower()):
            PARQUET_BASE_FOLDER = item_path
            break

# Проверяем БД
possible_db_paths = [
    os.path.join(SCRIPT_DIR, "telegram_messages.db"),  # Рядом со скриптом
    "telegram_messages.db",  # В текущей директории
    "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages.db",
    "/kaggle/input/telegram-messages/telegram_messages.db",  # На случай если запускается в Kaggle
]

db_path = None
for path in possible_db_paths:
    if os.path.exists(path):
        db_path = path
        break

print("=" * 60)
print("ПРОВЕРКА ВСЕХ PARQUET ФАЙЛОВ")
print("=" * 60)
print(f"Базовая папка: {PARQUET_BASE_FOLDER}")
if db_path:
    print(f"БД: {db_path}")
print("=" * 60)

# Ищем все parquet файлы (включая подпапки)
parquet_files = []

if os.path.exists(PARQUET_BASE_FOLDER):
    # Ищем файлы в основной папке и подпапках
    for root, dirs, files in os.walk(PARQUET_BASE_FOLDER):
        for file in files:
            if file.startswith("embeddings_") and file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                parquet_files.append(file_path)

# Сортируем по имени файла
parquet_files = sorted(parquet_files, key=lambda x: os.path.basename(x))

if not parquet_files:
    print("Parquet файлы не найдены")
    exit(1)

print(f"\nНайдено {len(parquet_files)} файлов\n")

# Подключаемся к БД
conn = None
if db_path:
    try:
        conn = sqlite3.connect(db_path)
    except:
        pass

# Статистика
results = []

for file_path in tqdm(parquet_files, desc="Проверка файлов"):
    f = os.path.basename(file_path)
    channel_name = f.replace("embeddings_", "").replace(".parquet", "")
    
    try:
        df = pl.read_parquet(file_path)
        
        # Проверяем эмбеддинги
        embeddings_list = df["embedding"].to_list()
        empty_count = 0
        
        for emb in embeddings_list:
            try:
                emb_array = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm < 0.01:  # Пустой эмбеддинг
                    empty_count += 1
            except:
                empty_count += 1
        
        # Проверяем в БД (если доступна)
        db_empty_count = None
        if conn:
            try:
                cursor = conn.cursor()
                msg_ids = df["message_id"].to_list()[:100]  # Проверяем первые 100 для примера
                if msg_ids:
                    placeholders = ','.join('?' * len(msg_ids))
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {channel_name} WHERE message_id IN ({placeholders}) AND (message IS NULL OR message = '' OR TRIM(message) = '')",
                        msg_ids
                    )
                    db_empty_count = cursor.fetchone()[0]
            except:
                pass
        
        results.append({
            'channel': channel_name,
            'total': len(df),
            'empty_by_norm': empty_count,
            'empty_percent': (empty_count / len(df) * 100) if len(df) > 0 else 0,
            'db_check': db_empty_count
        })
        
    except Exception as e:
        results.append({
            'channel': channel_name,
            'total': 0,
            'empty_by_norm': 0,
            'empty_percent': 0,
            'error': str(e)
        })

if conn:
    conn.close()

# Выводим результаты
print(f"\n{'=' * 80}")
print(f"{'Канал':<40} {'Всего':<10} {'Пустых':<10} {'%':<8} {'Статус':<12}")
print("=" * 80)

total_messages = 0
total_empty = 0

for r in results:
    channel = r['channel']
    total = r['total']
    empty = r['empty_by_norm']
    percent = r['empty_percent']
    
    if 'error' in r:
        status = f"{r['error'][:10]}"
    elif empty == 0:
        status = "OK"
    elif percent < 1:
        status = "Мало"
    else:
        status = f"{percent:.1f}%"
    
    print(f"{channel:<40} {total:>9,} {empty:>9,} {percent:>7.2f}% {status:<12}")
    
    total_messages += total
    total_empty += empty

print("=" * 80)
print(f"{'ИТОГО':<40} {total_messages:>9,} {total_empty:>9,} {(total_empty/total_messages*100) if total_messages > 0 else 0:>7.2f}%")

# Файлы с проблемами
problem_files = [r for r in results if r['empty_by_norm'] > 0]
if problem_files:
    print(f"\nФайлы с пустыми эмбеддингами ({len(problem_files)}):")
    for r in problem_files:
        print(f"   - {r['channel']}: {r['empty_by_norm']:,} пустых ({r['empty_percent']:.2f}%)")
    print(f"\nЗапустите clean_empty_embeddings.py для очистки")
else:
    print(f"\nВсе файлы в порядке - пустых эмбеддингов не найдено")
