"""
Конвертация .npy файлов с эмбеддингами в .parquet формат
для соответствия структуре файлов из "Большая модель эмбеддинг"
"""
import sqlite3
import polars as pl
import numpy as np
import os
from tqdm import tqdm

# Настройки
NPY_FOLDER = "Эмбендиннг жора"
OUTPUT_FOLDER = "Эмбендиннг жора parquet"
DB_PATH = "telegram_messages.db"

def convert_npy_to_parquet(npy_file: str, channel: str, db_conn):
    """
    Конвертирует один .npy файл в .parquet с метаданными из БД
    
    Args:
        npy_file: путь к .npy файлу
        channel: название канала (без префикса messages_)
        db_conn: подключение к SQLite БД
    """
    print(f"\nОбработка: {channel}")
    
    # 1. Загружаем эмбеддинги из .npy
    print("   Загружаю эмбеддинги из .npy...")
    embeddings_array = np.load(npy_file, mmap_mode='r')
    n_embeddings = embeddings_array.shape[0]
    emb_dim = embeddings_array.shape[1]
    print(f"   Загружено: {n_embeddings:,} эмбеддингов, размерность: {emb_dim}")
    
    # 2. Загружаем message_id и date из БД для непустых сообщений
    table_name = f"messages_{channel}"
    print(f"   Загружаю метаданные из БД (таблица: {table_name})...")
    
    query = f"""
        SELECT message_id, date
        FROM {table_name}
        WHERE LENGTH(message) > 0
        ORDER BY message_id
    """
    
    try:
        messages_df = pl.read_database(query, db_conn)
    except Exception as e:
        print(f"   Ошибка при чтении из БД: {e}")
        return False
    
    print(f"   Загружено сообщений из БД: {len(messages_df):,}")
    
    # 3. Проверяем соответствие количества
    if len(messages_df) != n_embeddings:
        print(f"   ВНИМАНИЕ: Несоответствие количества!")
        print(f"      БД: {len(messages_df):,} | Эмбеддинги: {n_embeddings:,}")
        print(f"      Разница: {abs(len(messages_df) - n_embeddings):,}")
        
        # Берем минимум для безопасности
        min_count = min(len(messages_df), n_embeddings)
        messages_df = messages_df.head(min_count)
        embeddings_array = embeddings_array[:min_count]
        print(f"   Использую первые {min_count:,} записей")
    
    # 4. Создаем DataFrame напрямую, используя более эффективный метод
    print("   Создаю DataFrame...")
    
    # Конвертируем эмбеддинги батчами для экономии памяти
    batch_size = 5000
    embedding_chunks = []
    
    for i in tqdm(range(0, n_embeddings, batch_size), desc="      ", leave=False):
        end_idx = min(i + batch_size, n_embeddings)
        batch = embeddings_array[i:end_idx]
        embedding_chunks.extend([emb.tolist() for emb in batch])
    
    # Создаем DataFrame
    df = pl.DataFrame({
        "message_id": messages_df["message_id"].to_list(),
        "date": messages_df["date"].to_list(),
        "embedding": embedding_chunks,
        "sentiment": [None] * len(messages_df)  # Пустой sentiment, как в "Большая модель"
    })
    
    # 6. Сохраняем в .parquet
    output_file = os.path.join(OUTPUT_FOLDER, f"embeddings_messages_{channel}.parquet")
    print(f"   Сохраняю в {output_file}...")
    df.write_parquet(output_file, compression="zstd")
    
    print(f"   Готово: {len(df):,} записей сохранено")
    return True


def main():
    print("=" * 80)
    print("КОНВЕРТАЦИЯ .NPY → .PARQUET")
    print("=" * 80)
    
    # Проверяем наличие папки с .npy файлами
    if not os.path.exists(NPY_FOLDER):
        print(f"Папка не найдена: {NPY_FOLDER}")
        return
    
    # Создаем выходную папку
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Выходная папка: {OUTPUT_FOLDER}")
    
    # Проверяем БД
    if not os.path.exists(DB_PATH):
        print(f"БД не найдена: {DB_PATH}")
        return
    
    # Подключаемся к БД
    print(f"Подключаюсь к БД: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    # Находим все .npy файлы
    npy_files = sorted([f for f in os.listdir(NPY_FOLDER) if f.endswith('.npy')])
    print(f"\nНайдено .npy файлов: {len(npy_files)}")
    
    if not npy_files:
        print("Не найдено .npy файлов!")
        conn.close()
        return
    
    # Обрабатываем каждый файл
    success_count = 0
    skipped_count = 0
    for npy_file in tqdm(npy_files, desc="Обработка файлов"):
        # Извлекаем название канала
        channel = npy_file.replace('.parquet.npy', '').replace('.npy', '')
        npy_path = os.path.join(NPY_FOLDER, npy_file)
        output_file = os.path.join(OUTPUT_FOLDER, f"embeddings_messages_{channel}.parquet")
        
        # Пропускаем, если файл уже существует
        if os.path.exists(output_file):
            print(f"\nПропускаю {channel} (файл уже существует)")
            skipped_count += 1
            continue
        
        try:
            if convert_npy_to_parquet(npy_path, channel, conn):
                success_count += 1
        except Exception as e:
            print(f"   Ошибка при обработке {npy_file}: {e}")
            import traceback
            traceback.print_exc()
    
    conn.close()
    
    print("\n" + "=" * 80)
    print(f"ГОТОВО!")
    print(f"   Успешно обработано: {success_count} из {len(npy_files)} файлов")
    print(f"   Результаты сохранены в: {OUTPUT_FOLDER}")
    print("=" * 80)


if __name__ == "__main__":
    main()
