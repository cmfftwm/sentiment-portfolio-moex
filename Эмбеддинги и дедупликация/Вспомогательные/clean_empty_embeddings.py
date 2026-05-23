"""
Очистка parquet файлов от эмбеддингов пустых сообщений
Проверяет все файлы и удаляет записи с пустыми эмбеддингами
"""

import polars as pl
import numpy as np
import os
import sqlite3
from pathlib import Path
from tqdm import tqdm

# Путь к папке с parquet файлами
# Автоматически ищет папку "Парквайт эмбенндинг" или можно указать вручную
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Папка находится в корне проекта, на уровень выше от скрипта
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET_BASE_FOLDER = os.path.join(PROJECT_ROOT, "Парквайт эмбенндинг")

# Если основной папки нет, ищем в текущей директории
if not os.path.exists(PARQUET_BASE_FOLDER):
    PARQUET_BASE_FOLDER = os.path.join(SCRIPT_DIR, ".")
    for item in os.listdir(SCRIPT_DIR):
        item_path = os.path.join(SCRIPT_DIR, item)
        if os.path.isdir(item_path) and ("парквайт" in item.lower() or "parquet" in item.lower() or "embedding" in item.lower()):
            PARQUET_BASE_FOLDER = item_path
            break

# Проверяем БД - пробуем разные возможные пути
possible_db_paths = [
    "telegram_messages.db",  # В текущей директории
    os.path.join(os.path.dirname(__file__), "telegram_messages.db"),  # Рядом со скриптом
    "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages.db",
    "/kaggle/input/telegram-messages/telegram_messages.db",  # На случай если запускается в Kaggle
]

db_path = None
for path in possible_db_paths:
    if os.path.exists(path):
        db_path = path
        break

USE_DB_CHECK = db_path is not None  # Использовать проверку через БД

print("=" * 60)
print("ОЧИСТКА PARQUET ФАЙЛОВ ОТ ПУСТЫХ ЭМБЕДДИНГОВ")
print("=" * 60)
print(f"Базовая папка: {PARQUET_BASE_FOLDER}")
if USE_DB_CHECK:
    print(f"БД для проверки: {db_path}")
else:
    print("БД не найдена, будет использоваться только проверка по норме эмбеддинга")
print("=" * 60)

# Ищем все parquet файлы (включая подпапки)
parquet_files = []
parquet_folders = []

# Проверяем основную папку
if os.path.exists(PARQUET_BASE_FOLDER):
    # Ищем файлы в основной папке
    for item in os.listdir(PARQUET_BASE_FOLDER):
        item_path = os.path.join(PARQUET_BASE_FOLDER, item)
        if os.path.isfile(item_path) and item.startswith("embeddings_") and item.endswith(".parquet"):
            parquet_files.append((item, item_path))
        elif os.path.isdir(item_path):
            # Проверяем подпапки
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                if os.path.isfile(sub_item_path) and sub_item.startswith("embeddings_") and sub_item.endswith(".parquet"):
                    parquet_files.append((sub_item, sub_item_path))

# Сортируем по имени файла
parquet_files = sorted(parquet_files, key=lambda x: x[0])
parquet_files = [f[1] for f in parquet_files]  # Оставляем только пути

if not parquet_files:
    print("Parquet файлы не найдены")
    exit(1)

print(f"\nНайдено {len(parquet_files)} parquet файлов\n")

# Подключаемся к БД, если доступна
conn = None
cursor = None
if USE_DB_CHECK:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("Подключено к БД для проверки пустых сообщений\n")
    except Exception as e:
        print(f"Не удалось подключиться к БД: {e}")
        USE_DB_CHECK = False

def is_empty_embedding(embedding, norm_threshold=0.01):
    """Проверяет, является ли эмбеддинг пустым (по норме)"""
    try:
        emb_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        # Нормализованные эмбеддинги должны иметь норму ~1.0
        # Пустые сообщения имеют очень маленькую норму
        return norm < norm_threshold
    except:
        return True  # Если ошибка - считаем пустым

def check_empty_in_db(table_name, message_id):
    """Проверяет в БД, является ли сообщение пустым"""
    if not USE_DB_CHECK or not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT message FROM {table_name} WHERE message_id = ?",
            (message_id,)
        )
        result = cursor.fetchone()
        if result:
            message = result[0]
            # Проверяем, пустое ли сообщение
            if message is None or message == '' or (message and message.strip() == ''):
                return True
            return False
    except:
        pass
    return None

# Обрабатываем каждый файл
total_removed = 0
total_checked = 0

for file_idx, file_path in enumerate(parquet_files, 1):
    f = os.path.basename(file_path)
    channel_name = f.replace("embeddings_", "").replace(".parquet", "")
    
    print(f"\n[{file_idx}/{len(parquet_files)}] Обработка: {channel_name}")
    print("-" * 60)
    
    try:
        # Читаем файл
        df = pl.read_parquet(file_path)
        original_count = len(df)
        print(f"   Исходное количество: {original_count:,}")
        
        if original_count == 0:
            print("   Файл пустой, пропускаю")
            continue
        
        # Метод 1: Проверка по норме эмбеддинга
        print("   Проверяю эмбеддинги по норме...")
        embeddings_list = df["embedding"].to_list()
        norms = []
        empty_by_norm = []
        
        for idx, emb in enumerate(tqdm(embeddings_list, desc="      Вычисляю нормы", leave=False)):
            norm = np.linalg.norm(np.array(emb, dtype=np.float32))
            norms.append(norm)
            if norm < 0.01:  # Порог для пустых эмбеддингов
                empty_by_norm.append(idx)
        
        # Метод 2: Проверка через БД (если доступна) - БАТЧАМИ для скорости
        empty_by_db = []
        if USE_DB_CHECK and conn and cursor:
            print("   Проверяю через БД (батчами)...")
            msg_ids = df["message_id"].to_list()
            
            # Проверяем батчами по 1000 для эффективности
            for batch_start in tqdm(range(0, len(msg_ids), 1000), desc="      Батчи БД", leave=False):
                batch_end = min(batch_start + 1000, len(msg_ids))
                batch_ids = msg_ids[batch_start:batch_end]
                
                try:
                    placeholders = ','.join('?' * len(batch_ids))
                    query = f"""
                        SELECT message_id 
                        FROM {channel_name}
                        WHERE message_id IN ({placeholders})
                        AND (message IS NULL OR message = '' OR TRIM(message) = '')
                    """
                    cursor.execute(query, batch_ids)
                    batch_empty_ids = {row[0] for row in cursor.fetchall()}
                    
                    # Находим индексы пустых в текущем батче
                    for local_idx, msg_id in enumerate(batch_ids):
                        if msg_id in batch_empty_ids:
                            global_idx = batch_start + local_idx
                            empty_by_db.append(global_idx)
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        print(f"      Таблица {channel_name} не найдена в БД")
                    break
                except Exception as e:
                    print(f"      Ошибка при проверке батча: {e}")
                    break
        
        # Объединяем результаты
        # Приоритет БД - если БД говорит что пустое, то удаляем
        if USE_DB_CHECK and empty_by_db:
            # Используем БД как основной источник (более надежно)
            empty_indices = set(empty_by_db)
            # Добавляем также те, что по норме (на случай если БД недоступна для некоторых)
            empty_indices.update(empty_by_norm)
        else:
            # Только по норме (если БД недоступна)
            empty_indices = set(empty_by_norm)
        
        empty_count = len(empty_indices)
        
        if empty_count > 0:
            print(f"   Найдено пустых эмбеддингов: {empty_count:,}")
            if USE_DB_CHECK:
                print(f"      По БД (основной метод): {len(empty_by_db):,}")
                print(f"      По норме (дополнительно): {len(empty_by_norm):,}")
            else:
                print(f"      По норме: {len(empty_by_norm):,}")
            
            # Удаляем пустые
            indices_to_keep = [i for i in range(len(df)) if i not in empty_indices]
            df_cleaned = df[indices_to_keep]
            
            print(f"   После очистки: {len(df_cleaned):,}")
            print(f"   Удалено: {empty_count:,}")
            
            # Сохраняем очищенный файл
            print(f"   Сохраняю очищенный файл...")
            df_cleaned.write_parquet(file_path, compression="snappy")
            
            # Показываем размер
            new_size = os.path.getsize(file_path) / (1024**2)
            print(f"   Новый размер: {new_size:.2f} MB")
            
            total_removed += empty_count
        else:
            print(f"   Пустых эмбеддингов не найдено")
        
        total_checked += original_count
        
    except Exception as e:
        print(f"   Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()

if conn:
    if cursor:
        cursor.close()
    conn.close()

print(f"\n{'=' * 60}")
print(f"ОЧИСТКА ЗАВЕРШЕНА")
print(f"{'=' * 60}")
print(f"Итоговая статистика:")
print(f"   Проверено файлов: {len(parquet_files)}")
print(f"   Проверено сообщений: {total_checked:,}")
print(f"   Удалено пустых: {total_removed:,}")
if total_checked > 0:
    percentage = (total_removed / total_checked) * 100
    print(f"   Процент удаленных: {percentage:.2f}%")
print(f"{'=' * 60}")
