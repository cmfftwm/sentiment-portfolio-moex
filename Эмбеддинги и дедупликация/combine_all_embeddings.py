"""
Объединение всех parquet файлов с эмбеддингами в один общий файл
"""

import polars as pl
import os
from pathlib import Path
from tqdm import tqdm

# Автоматическое определение путей
SCRIPT_DIR = Path(__file__).parent.absolute()

# Путь к папке с parquet файлами
PARQUET_BASE_FOLDER = SCRIPT_DIR / "Эмбендинги за по года"

# Путь для сохранения общего файла
OUTPUT_FILE = SCRIPT_DIR / "all_embeddings.parquet"

print("=" * 80)
print("ОБЪЕДИНЕНИЕ ВСЕХ PARQUET ФАЙЛОВ С ЭМБЕДДИНГАМИ")
print("=" * 80)
print(f"Папка с parquet: {PARQUET_BASE_FOLDER}")
print(f"Выходной файл: {OUTPUT_FILE}")
print("=" * 80)
print()

if not PARQUET_BASE_FOLDER.exists():
    print(f"Папка с parquet файлами не найдена: {PARQUET_BASE_FOLDER}")
    exit(1)

# Находим все parquet файлы (включая подпапки)
parquet_files = []
for root, dirs, files in os.walk(PARQUET_BASE_FOLDER):
    for file in files:
        if file.startswith("embeddings_") and file.endswith(".parquet"):
            full_path = os.path.join(root, file)
            channel_name = file.replace("embeddings_", "").replace(".parquet", "")
            parquet_files.append((full_path, channel_name))

parquet_files = sorted(parquet_files, key=lambda x: x[1])

if not parquet_files:
    print("Parquet файлы не найдены")
    exit(1)

print(f"Найдено {len(parquet_files)} parquet файлов\n")

# Собираем все данные
all_dataframes = []
total_rows = 0

print("Чтение файлов...")
for file_path, channel_name in tqdm(parquet_files, desc="Обработка"):
    try:
        df = pl.read_parquet(file_path)
        
        # Добавляем колонку channel, если её нет
        if "channel" not in df.columns:
            df = df.with_columns(pl.lit(channel_name).alias("channel"))
        else:
            # Если есть, но значения пустые, заполняем
            df = df.with_columns(
                pl.when(pl.col("channel").is_null())
                .then(pl.lit(channel_name))
                .otherwise(pl.col("channel"))
                .alias("channel")
            )
        
        # Проверяем наличие всех необходимых колонок
        required_cols = ["message_id", "date", "embedding", "channel"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   {channel_name}: отсутствуют колонки {missing_cols}")
            continue
        
        # Переупорядочиваем колонки: channel, message_id, date, embedding
        # Убираем sentiment, если он есть
        df = df.select(["channel", "message_id", "date", "embedding"])
        
        all_dataframes.append(df)
        total_rows += len(df)
        
    except Exception as e:
        print(f"   Ошибка при чтении {channel_name}: {e}")
        continue

if not all_dataframes:
    print("Не удалось прочитать ни одного файла")
    exit(1)

print(f"\nПрочитано {len(all_dataframes)} файлов, всего {total_rows:,} записей\n")

# Объединяем все DataFrame
print("Объединение всех данных...")
combined_df = pl.concat(all_dataframes)

print(f"   Исходное количество записей: {len(combined_df):,}")

# Сортируем по дате
print("Сортировка по дате...")
combined_df = combined_df.sort("date")

# Сохраняем в один файл
print(f"\nСохранение в {OUTPUT_FILE}...")
combined_df.write_parquet(
    str(OUTPUT_FILE),
    compression="snappy"
)

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
file_size_gb = file_size_mb / 1024

print("\n" + "=" * 80)
print("ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО")
print("=" * 80)
print(f"Итоговая статистика:")
print(f"   Обработано файлов: {len(all_dataframes)}")
print(f"   Всего записей: {len(combined_df):,}")
print(f"   Размер файла: {file_size_mb:.1f} MB ({file_size_gb:.2f} GB)")
print(f"   Путь: {OUTPUT_FILE}")
print("=" * 80)

# Показываем статистику по каналам
print("\nСтатистика по каналам:")
channel_stats = combined_df.group_by("channel").agg([
    pl.len().alias("count")
]).sort("count", descending=True)

for row in channel_stats.iter_rows(named=True):
    print(f"   {row['channel']:<45} {row['count']:>11,}")

print("\nГотово!")
