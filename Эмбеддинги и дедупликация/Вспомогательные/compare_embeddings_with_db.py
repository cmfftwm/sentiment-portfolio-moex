"""
Сравнение количества эмбеддингов в parquet файлах с количеством сообщений в БД
"""

import polars as pl
import sqlite3
import os
from pathlib import Path
from tqdm import tqdm

# Автоматическое определение путей
SCRIPT_DIR = Path(__file__).parent.absolute()
# Папка находится в корне проекта, на уровень выше от скрипта
PROJECT_ROOT = SCRIPT_DIR.parent

# Путь к папке с parquet файлами
PARQUET_BASE_FOLDER = PROJECT_ROOT / "Парквайт эмбенндинг"
if not PARQUET_BASE_FOLDER.exists():
    # Пробуем найти папку
    for item in SCRIPT_DIR.iterdir():
        if item.is_dir() and ("parquet" in item.name.lower() or "embedding" in item.name.lower()):
            PARQUET_BASE_FOLDER = item
            break

# Путь к БД
DB_PATH = PROJECT_ROOT / "telegram_messages.db"
if not DB_PATH.exists():
    # Пробуем найти БД
    for item in SCRIPT_DIR.iterdir():
        if item.is_file() and item.name.endswith(".db") and "telegram" in item.name.lower():
            DB_PATH = item
            break

print("=" * 80)
print("СРАВНЕНИЕ ЭМБЕДДИНГОВ С БД")
print("=" * 80)
print(f"Папка с parquet: {PARQUET_BASE_FOLDER}")
print(f"БД: {DB_PATH}")
print("=" * 80)
print()

if not PARQUET_BASE_FOLDER.exists():
    print(f"Папка с parquet файлами не найдена: {PARQUET_BASE_FOLDER}")
    exit(1)

if not DB_PATH.exists():
    print(f"База данных не найдена: {DB_PATH}")
    exit(1)

# Находим все parquet файлы (включая подпапки)
parquet_files = []
for root, dirs, files in os.walk(PARQUET_BASE_FOLDER):
    for file in files:
        if file.startswith("embeddings_") and file.endswith(".parquet"):
            parquet_files.append((root, file))

parquet_files = sorted(parquet_files, key=lambda x: x[1])

if not parquet_files:
    print("Parquet файлы не найдены")
    exit(1)

print(f"Найдено {len(parquet_files)} parquet файлов\n")

# Подключаемся к БД
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Получаем список всех таблиц-каналов
cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'message_embeddings';"
)
all_tables = [row[0] for row in cursor.fetchall()]

print("Сравнение по каналам:\n")
print(f"{'Канал':<45} {'Parquet':<12} {'БД (непустые)':<15} {'Разница':<12} {'Статус':<10}")
print("=" * 100)

total_parquet = 0
total_db = 0
total_diff = 0
channels_with_diff = []

for root, file in tqdm(parquet_files, desc="Обработка файлов"):
    file_path = os.path.join(root, file)
    channel_name = file.replace("embeddings_", "").replace(".parquet", "")
    
    # Читаем parquet
    try:
        df = pl.read_parquet(file_path)
        parquet_count = len(df)
        total_parquet += parquet_count
    except Exception as e:
        print(f"{channel_name:<45} {'ERROR':<12} {'-':<15} {'-':<12} {str(e)[:20]:<10}")
        continue
    
    # Подсчитываем непустые сообщения в БД
    db_count = 0
    if channel_name in all_tables:
        try:
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM {channel_name}
                WHERE message IS NOT NULL 
                AND message != '' 
                AND TRIM(message) != ''
            """)
            db_count = cursor.fetchone()[0]
            total_db += db_count
        except Exception as e:
            status = f"Ошибка БД: {str(e)[:15]}"
            print(f"{channel_name:<45} {parquet_count:>11,} {'-':<15} {'-':<12} {status:<10}")
            continue
    else:
        status = "Нет в БД"
        print(f"{channel_name:<45} {parquet_count:>11,} {'-':<15} {'-':<12} {status:<10}")
        continue
    
    # Сравниваем
    diff = parquet_count - db_count
    total_diff += abs(diff)
    
    if diff == 0:
        status = "OK"
    elif diff > 0:
        status = f"+{diff:,}"
        channels_with_diff.append((channel_name, diff, "больше в parquet"))
    else:
        status = f"{diff:,}"
        channels_with_diff.append((channel_name, abs(diff), "меньше в parquet"))
    
    print(f"{channel_name:<45} {parquet_count:>11,} {db_count:>14,} {diff:>11,} {status:<10}")

print("=" * 100)
print(f"{'ИТОГО':<45} {total_parquet:>11,} {total_db:>14,} {total_diff:>11,}")

print("\n" + "=" * 80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 80)
print(f"Всего эмбеддингов в parquet: {total_parquet:,}")
print(f"Всего непустых сообщений в БД: {total_db:,}")
print(f"Разница: {total_parquet - total_db:,}")
if total_db > 0:
    coverage = (total_parquet / total_db) * 100
    print(f"Покрытие: {coverage:.2f}%")
print()

if channels_with_diff:
    print("Каналы с расхождениями:")
    for channel, diff, direction in channels_with_diff:
        print(f"   - {channel}: {diff:,} {direction}")
else:
    print("Все каналы совпадают!")

# Проверяем, есть ли каналы в БД, но нет в parquet
parquet_channels = {f[1].replace("embeddings_", "").replace(".parquet", "") for f in parquet_files}
db_channels = set(all_tables)
missing_in_parquet = db_channels - parquet_channels
missing_in_db = parquet_channels - db_channels

if missing_in_parquet:
    print(f"\nКаналы в БД, но нет в parquet ({len(missing_in_parquet)}):")
    for ch in sorted(missing_in_parquet):
        try:
            # Проверяем, есть ли колонка message
            cursor.execute(f"PRAGMA table_info({ch})")
            columns = [row[1] for row in cursor.fetchall()]
            if 'message' in columns:
                cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM {ch}
                    WHERE message IS NOT NULL 
                    AND message != '' 
                    AND TRIM(message) != ''
                """)
                count = cursor.fetchone()[0]
                print(f"   - {ch}: {count:,} сообщений")
            else:
                # Если нет колонки message, просто считаем все записи
                cursor.execute(f"SELECT COUNT(*) FROM {ch}")
                count = cursor.fetchone()[0]
                print(f"   - {ch}: {count:,} записей (нет колонки message)")
        except Exception as e:
            print(f"   - {ch}: ошибка при проверке ({str(e)[:30]})")

if missing_in_db:
    print(f"\nКаналы в parquet, но нет в БД ({len(missing_in_db)}):")
    for ch in sorted(missing_in_db):
        print(f"   - {ch}")

conn.close()

print("\n" + "=" * 80)
print("Проверка завершена")
print("=" * 80)
