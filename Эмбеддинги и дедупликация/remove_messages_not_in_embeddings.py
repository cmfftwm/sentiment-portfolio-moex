"""
Скрипт для удаления из telegram_messages.db сообщений, которых нет в all_embeddings.parquet

ЛОГИКА РАБОТЫ:
- Скрипт удаляет сообщения из таблиц каналов (например, messages_headlines_quants), 
  для которых НЕТ соответствующей записи в all_embeddings.parquet
- То есть удаляются сообщения, для которых не был создан эмбеддинг
- Также удаляются соответствующие записи из таблицы message_embeddings

ПАРАМЕТРЫ:
- DRY_RUN = True: только проверка, без удаления (показывает что будет удалено)
- DRY_RUN = False: реальное удаление
- ONLY_PROCESS_CHANNELS_IN_PARQUET = True: удалять сообщения ТОЛЬКО из каналов, 
  которые ЕСТЬ в all_embeddings.parquet (безопасный режим)
- ONLY_PROCESS_CHANNELS_IN_PARQUET = False: удалять из ВСЕХ каналов, включая те, 
  которых нет в parquet (они будут удалены полностью)

РЕКОМЕНДАЦИЯ:
- Используйте ONLY_PROCESS_CHANNELS_IN_PARQUET = True, если хотите удалить только 
  те сообщения, для которых нет эмбеддинга, но канал присутствует в parquet
- Используйте ONLY_PROCESS_CHANNELS_IN_PARQUET = False, если хотите удалить все 
  сообщения из каналов, которых вообще нет в parquet
"""

import sqlite3
import polars as pl
from pathlib import Path
from tqdm import tqdm
import sys

# Автоматическое определение путей
SCRIPT_DIR = Path(__file__).parent.absolute()

# Путь к файлу с эмбеддингами
EMBEDDINGS_FILE = SCRIPT_DIR / "all_embeddings.parquet"

# Путь к БД
DB_PATH = SCRIPT_DIR / "telegram_messages.db"

# РЕЖИМ ПРОВЕРКИ (без удаления) - установите False для реального удаления
DRY_RUN = False

# ОБРАБАТЫВАТЬ ТОЛЬКО КАНАЛЫ ИЗ PARQUET
# True: удалять сообщения только из каналов, которые ЕСТЬ в all_embeddings.parquet
# False: удалять из всех каналов (включая те, которых нет в parquet - они будут удалены полностью)
ONLY_PROCESS_CHANNELS_IN_PARQUET = False

print("=" * 80)
if DRY_RUN:
    print("РЕЖИМ ПРОВЕРКИ (DRY RUN) - удаление НЕ будет выполнено")
else:
    print("УДАЛЕНИЕ СООБЩЕНИЙ ИЗ БД, КОТОРЫХ НЕТ В ALL_EMBEDDINGS.PARQUET")
print("=" * 80)
print(f"Файл с эмбеддингами: {EMBEDDINGS_FILE}")
print(f"База данных: {DB_PATH}")
print(f"Режим: {'ПРОВЕРКА (без удаления)' if DRY_RUN else 'РЕАЛЬНОЕ УДАЛЕНИЕ'}")
print(f"Обработка каналов: {'ТОЛЬКО из каналов в parquet' if ONLY_PROCESS_CHANNELS_IN_PARQUET else 'ИЗ ВСЕХ каналов (включая отсутствующие в parquet)'}")
print("=" * 80)
print()

# Проверяем наличие файлов
if not EMBEDDINGS_FILE.exists():
    print(f"Файл с эмбеддингами не найден: {EMBEDDINGS_FILE}")
    exit(1)

if not DB_PATH.exists():
    print(f"База данных не найдена: {DB_PATH}")
    exit(1)

# Читаем файл с эмбеддингами
print("Загружаю all_embeddings.parquet...")
try:
    df_embeddings = pl.read_parquet(EMBEDDINGS_FILE)
    print(f"   Загружено записей: {len(df_embeddings):,}")
except Exception as e:
    print(f"Ошибка при чтении parquet файла: {e}")
    exit(1)

# Проверяем наличие необходимых колонок
required_cols = ["channel", "message_id"]
missing_cols = [col for col in required_cols if col not in df_embeddings.columns]
if missing_cols:
    print(f"В parquet файле отсутствуют колонки: {missing_cols}")
    exit(1)

# Создаем множество пар (channel, message_id) из parquet
print("\nСоздаю множество пар (channel, message_id) из parquet...")
embeddings_set = set()
channels_in_parquet = set()
for row in df_embeddings.iter_rows(named=True):
    channel = str(row["channel"])
    message_id = int(row["message_id"])
    embeddings_set.add((channel, message_id))
    channels_in_parquet.add(channel)

print(f"   Найдено уникальных пар: {len(embeddings_set):,}")
print(f"   Уникальных каналов в parquet: {len(channels_in_parquet)}")

# Подключаемся к БД
print("\nПодключаюсь к базе данных...")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Получаем список всех таблиц-каналов (исключаем служебные таблицы)
cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'message_embeddings';"
)
all_tables = [row[0] for row in cursor.fetchall()]

print(f"   Найдено таблиц-каналов: {len(all_tables)}")

# Фильтруем таблицы, если нужно обрабатывать только каналы из parquet
if ONLY_PROCESS_CHANNELS_IN_PARQUET:
    tables_to_process = [t for t in all_tables if t in channels_in_parquet]
    skipped_tables = [t for t in all_tables if t not in channels_in_parquet]
    print(f"   Будет обработано каналов: {len(tables_to_process)} (только из parquet)")
    if skipped_tables:
        print(f"   Пропущено каналов (нет в parquet): {len(skipped_tables)}")
        if len(skipped_tables) <= 10:
            for t in skipped_tables:
                print(f"      - {t}")
        else:
            for t in skipped_tables[:10]:
                print(f"      - {t}")
            print(f"      ... и еще {len(skipped_tables) - 10} каналов")
else:
    tables_to_process = all_tables
    print(f"   Будет обработано каналов: {len(tables_to_process)} (все каналы)")

# Проверяем соответствие имен таблиц и каналов в parquet
print("\nПроверка соответствия имен таблиц и каналов...")
tables_not_in_parquet = [t for t in all_tables if t not in channels_in_parquet]
if tables_not_in_parquet:
    print(f"   Найдено таблиц в БД, которых нет в parquet: {len(tables_not_in_parquet)}")
    if ONLY_PROCESS_CHANNELS_IN_PARQUET:
        print("   (эти таблицы будут ПРОПУЩЕНЫ, так как ONLY_PROCESS_CHANNELS_IN_PARQUET = True)")
    else:
        print("   (сообщения из этих таблиц будут УДАЛЕНЫ полностью, так как их нет в parquet)")
    if len(tables_not_in_parquet) <= 10:
        for t in tables_not_in_parquet:
            print(f"      - {t}")
    else:
        for t in tables_not_in_parquet[:10]:
            print(f"      - {t}")
        print(f"      ... и еще {len(tables_not_in_parquet) - 10} таблиц")

channels_not_in_db = [c for c in channels_in_parquet if c not in all_tables]
if channels_not_in_db:
    print(f"   Найдено каналов в parquet, которых нет в БД: {len(channels_not_in_db)}")
    if len(channels_not_in_db) <= 10:
        for c in channels_not_in_db:
            print(f"      - {c}")
    else:
        for c in channels_not_in_db[:10]:
            print(f"      - {c}")
        print(f"      ... и еще {len(channels_not_in_db) - 10} каналов")

# Статистика
total_deleted = 0
channels_with_deletions = []
messages_to_delete_details = []  # Для детального вывода
tables_to_drop = []  # Таблицы, которые нужно удалить полностью

print("\n" + "=" * 80)
if DRY_RUN:
    print("ПРОВЕРКА: какие сообщения будут удалены")
else:
    print("УДАЛЕНИЕ СООБЩЕНИЙ")
print("=" * 80)

# Обрабатываем каждую таблицу-канал
for table_name in tqdm(tables_to_process, desc="Обработка каналов"):
    try:
        # Получаем все message_id из таблицы канала
        cursor.execute(f"SELECT message_id FROM {table_name}")
        db_message_ids = {row[0] for row in cursor.fetchall()}
        
        if not db_message_ids:
            continue
        
        # Находим message_id, которых нет в parquet
        ids_to_delete = []
        for msg_id in db_message_ids:
            if (table_name, msg_id) not in embeddings_set:
                ids_to_delete.append(msg_id)
        
        if not ids_to_delete:
            continue
        
        deleted_count = len(ids_to_delete)
        total_deleted += deleted_count
        channels_with_deletions.append((table_name, deleted_count))
        
        # Сохраняем примеры для детального вывода (первые 5)
        sample_ids = sorted(ids_to_delete)[:5]
        messages_to_delete_details.append({
            'table': table_name,
            'count': deleted_count,
            'sample_ids': sample_ids
        })
        
        # Удаляем сообщения из таблицы канала (только если не DRY_RUN)
        # Разбиваем на батчи, чтобы избежать ошибки "too many SQL variables"
        # SQLite ограничивает количество параметров (обычно 999 или 32766)
        if not DRY_RUN:
            BATCH_SIZE = 500  # Безопасный размер батча
            actual_deleted = 0
            
            for i in range(0, len(ids_to_delete), BATCH_SIZE):
                batch = ids_to_delete[i:i + BATCH_SIZE]
                placeholders = ",".join("?" * len(batch))
                cursor.execute(
                    f"DELETE FROM {table_name} WHERE message_id IN ({placeholders})",
                    batch
                )
                actual_deleted += cursor.rowcount
            
            # Проверяем, остались ли сообщения в таблице после удаления
            # Если канала нет в parquet и таблица пуста - помечаем для удаления
            if table_name not in channels_in_parquet:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                remaining_count = cursor.fetchone()[0]
                if remaining_count == 0:
                    tables_to_drop.append(table_name)
        
        # Выводим информацию о прогрессе
        action = "будет удалено" if DRY_RUN else "удалено"
        print(f"   {'' if DRY_RUN else ''} {table_name}: {action} {deleted_count:,} сообщений")
        
        # В режиме проверки также показываем, какие таблицы будут удалены
        if DRY_RUN and table_name not in channels_in_parquet:
            # Проверяем, сколько сообщений будет удалено
            if deleted_count == len(db_message_ids):
                tables_to_drop.append(table_name)
    
    except Exception as e:
        print(f"   Ошибка при обработке {table_name}: {e}")
        continue

# Также удаляем записи из таблицы message_embeddings, которых нет в parquet
print("\nОчистка таблицы message_embeddings...")
try:
    cursor.execute("SELECT channel, message_id FROM message_embeddings")
    all_embeddings_db = cursor.fetchall()
    
    embeddings_to_delete = []
    for channel, message_id in all_embeddings_db:
        if (str(channel), int(message_id)) not in embeddings_set:
            embeddings_to_delete.append((channel, message_id))
    
    if embeddings_to_delete:
        deleted_embeddings_count = len(embeddings_to_delete)
        if not DRY_RUN:
            for channel, message_id in tqdm(embeddings_to_delete, desc="   Удаление из message_embeddings"):
                cursor.execute(
                    "DELETE FROM message_embeddings WHERE channel = ? AND message_id = ?",
                    (channel, message_id)
                )
        
        action = "будет удалено" if DRY_RUN else "удалено"
        print(f"   {'' if DRY_RUN else ''} {action.capitalize()} записей из message_embeddings: {deleted_embeddings_count:,}")
    else:
        print("   В message_embeddings нет записей для удаления")

except Exception as e:
    print(f"   Ошибка при очистке message_embeddings: {e}")

# Удаляем пустые таблицы каналов, которых нет в parquet
if tables_to_drop:
    print(f"\nУдаление пустых таблиц каналов (которых нет в parquet)...")
    for table_name in tqdm(tables_to_drop, desc="   Удаление таблиц"):
        try:
            if not DRY_RUN:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            action = "будет удалена" if DRY_RUN else "удалена"
            print(f"   {'' if DRY_RUN else ''} Таблица {table_name}: {action}")
        except Exception as e:
            print(f"   Ошибка при удалении таблицы {table_name}: {e}")
else:
    print("\nПустых таблиц для удаления не найдено")

# Коммитим изменения (только если не DRY_RUN)
if not DRY_RUN:
    print("\nСохранение изменений...")
    conn.commit()
    print("   Изменения сохранены")
else:
    print("\nРежим проверки: изменения НЕ сохранены")

# Выводим итоговую статистику
print("\n" + "=" * 80)
print("ИТОГОВАЯ СТАТИСТИКА")
print("=" * 80)
action = "будет удалено" if DRY_RUN else "удалено"
print(f"Всего {action} сообщений из таблиц каналов: {total_deleted:,}")
print(f"Каналов с удалениями: {len(channels_with_deletions)}")

if channels_with_deletions:
    print("\nДетализация по каналам:")
    channels_with_deletions.sort(key=lambda x: x[1], reverse=True)
    for table_name, count in channels_with_deletions:
        print(f"   - {table_name}: {count:,} сообщений")
    
    # Показываем примеры удаляемых message_id
    if messages_to_delete_details:
        print("\nПримеры message_id, которые будут удалены (первые 5 для каждого канала):")
        for detail in messages_to_delete_details[:10]:  # Показываем первые 10 каналов
            print(f"   - {detail['table']}: {detail['sample_ids']}")

if tables_to_drop:
    action = "будут удалены" if DRY_RUN else "удалены"
    print(f"\nТаблицы, которые {action} (каналы, которых нет в parquet, после удаления всех сообщений):")
    for table_name in sorted(tables_to_drop):
        print(f"   - {table_name}")

if DRY_RUN:
    print("\n" + "=" * 80)
    print("ВНИМАНИЕ: Это был режим проверки (DRY RUN)")
    print("   Для реального удаления установите DRY_RUN = False в скрипте")
    print("=" * 80)

print("\n" + "=" * 80)
print("Обработка завершена")
print("=" * 80)

# Закрываем соединение
conn.close()
