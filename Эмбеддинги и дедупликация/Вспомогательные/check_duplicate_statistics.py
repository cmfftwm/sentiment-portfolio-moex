"""
Проверка статистики дубликатов: сколько из одного канала, сколько из разных
"""
import sys
from pathlib import Path

try:
    import polars as pl
except ImportError:
    print("Не установлен polars. Установите: pip install polars")
    sys.exit(1)

# Пути
SCRIPT_DIR = Path(__file__).parent.absolute()
DUPS_FILE = SCRIPT_DIR / "all_embeddings_duplicates.parquet"

if not DUPS_FILE.exists():
    print(f"Файл {DUPS_FILE} не найден")
    sys.exit(1)

print("=" * 80)
print("СТАТИСТИКА ДУБЛИКАТОВ")
print("=" * 80)

# Читаем файл дубликатов
dups_df = pl.read_parquet(DUPS_FILE)

print(f"\nВсего дубликатов: {len(dups_df):,}")

# Дубликаты из одного канала
same_channel = dups_df.filter(pl.col("channel") == pl.col("base_channel"))
print(f"Дубликаты в одном канале: {len(same_channel):,} ({100*len(same_channel)/len(dups_df):.1f}%)")

# Дубликаты из разных каналов
cross_channel = dups_df.filter(pl.col("channel") != pl.col("base_channel"))
print(f"Дубликаты из разных каналов: {len(cross_channel):,} ({100*len(cross_channel)/len(dups_df):.1f}%)")

if len(cross_channel) > 0:
    print("\nНайдены межканальные дубликаты!")
    print("\nПримеры межканальных дубликатов (первые 10):")
    print("-" * 80)
    for i, row in enumerate(cross_channel.head(10).iter_rows(named=True), 1):
        print(f"{i}. {row['base_channel']} #{row['base_message_id']} ↔ {row['channel']} #{row['message_id']}")
        print(f"   Сходство: {row['sim']:.4f} | Разница: {row.get('dt', 0):.2f} часов")
else:
    print("\nМежканальные дубликаты не найдены!")
    print("\nВозможные причины:")
    print("   1. Новости действительно не дублируются между каналами в пределах 24 часов")
    print("   2. Порог косинусного сходства (0.96) слишком высокий")
    print("   3. Разные каналы формулируют одну новость по-разному")
    print("\nРекомендации:")
    print("   - Попробуйте снизить COSINE_THRESHOLD до 0.94-0.95")
    print("   - Увеличьте WINDOW_HOURS до 48-72 часов")

# Статистика по каналам (топ-10 каналов с наибольшим количеством дубликатов)
print("\n" + "=" * 80)
print("Топ-10 каналов с наибольшим количеством дубликатов:")
print("=" * 80)

channel_dup_counts = dups_df.group_by("channel").agg([
    pl.len().alias("count")
]).sort("count", descending=True).head(10)

for row in channel_dup_counts.iter_rows(named=True):
    print(f"   {row['channel']:<45} {row['count']:>6,} дубликатов")

print("\nГотово!")
