"""
Проверка и работа с parquet файлами эмбеддингов
"""

import polars as pl
import os
from pathlib import Path

# Путь к папке с parquet файлами
PARQUET_FOLDER = "/kaggle/working"  # Измените на вашу папку

# Если папка не указана, ищем в текущей директории
if not os.path.exists(PARQUET_FOLDER):
    PARQUET_FOLDER = "."
    # Пробуем найти папку с parquet файлами
    for item in os.listdir("."):
        if os.path.isdir(item) and "parquet" in item.lower() or "embedding" in item.lower():
            PARQUET_FOLDER = item
            break

print("=" * 60)
print("ПРОВЕРКА PARQUET ФАЙЛОВ ЭМБЕДДИНГОВ")
print("=" * 60)
print(f"Папка: {PARQUET_FOLDER}")
print()

# Ищем все parquet файлы
parquet_files = sorted([
    f for f in os.listdir(PARQUET_FOLDER) 
    if f.startswith("embeddings_") and f.endswith(".parquet")
])

if not parquet_files:
    print("Parquet файлы не найдены")
    print(f"   Проверяемая папка: {os.path.abspath(PARQUET_FOLDER)}")
    # Показываем что есть в папке
    all_files = [f for f in os.listdir(PARQUET_FOLDER) if f.endswith('.parquet')]
    if all_files:
        print(f"   Найдены другие parquet файлы: {all_files[:5]}")
    exit(1)

print(f"Найдено {len(parquet_files)} parquet файлов\n")

# Статистика по каждому файлу
total_messages = 0
total_size = 0

print(f"{'Канал':<45} {'Сообщений':<12} {'Размер (MB)':<12} {'Статус':<10}")
print("=" * 80)

for f in parquet_files:
    file_path = os.path.join(PARQUET_FOLDER, f)
    channel_name = f.replace("embeddings_", "").replace(".parquet", "")
    
    try:
        df = pl.read_parquet(file_path)
        file_size = os.path.getsize(file_path) / (1024**2)
        
        # Проверяем структуру
        has_all_cols = all(col in df.columns for col in ['message_id', 'date', 'embedding', 'sentiment'])
        status = "OK" if has_all_cols else "Структура"
        
        print(f"{channel_name:<45} {len(df):>11,} {file_size:>11.1f} {status:<10}")
        
        total_messages += len(df)
        total_size += file_size
        
    except Exception as e:
        print(f"{channel_name:<45} {'ERROR':<12} {'-':<12} {str(e)[:20]:<10}")

print("=" * 80)
print(f"{'ИТОГО':<45} {total_messages:>11,} {total_size:>11.1f}")
print()

# Проверяем, какие каналы обработаны
print("Список обработанных каналов:")
for i, f in enumerate(parquet_files, 1):
    channel_name = f.replace("embeddings_", "").replace(".parquet", "")
    print(f"   {i:2}. {channel_name}")

print(f"\nВсего обработано: {len(parquet_files)} каналов")
print(f"   Всего сообщений: {total_messages:,}")
print(f"   Общий размер: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
