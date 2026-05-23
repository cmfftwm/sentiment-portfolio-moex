"""
Дедупликация эмбеддингов из parquet файлов
Работает с отдельными parquet файлами по каналам или объединенным файлом
"""

import polars as pl
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

WINDOW_HOURS = 24        # временное окно для поиска дубликатов
COSINE_THRESHOLD = 0.96 # порог похожести для дублей (повышен до 0.96 для уменьшения ложных срабатываний)

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя векторами"""
    # Правильная формула косинусного сходства: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
    # Это гарантирует результат в диапазоне [-1, 1], даже если векторы не нормализованы
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(np.dot(v1, v2) / denom)

def deduplicate_parquet_file(parquet_path, output_path=None, duplicates_path=None):
    """
    Дедупликация одного parquet файла
    
    Args:
        parquet_path: путь к parquet файлу
        output_path: путь для сохранения уникальных записей (если None, перезаписывает исходный)
        duplicates_path: путь для сохранения дубликатов (если None, создается автоматически)
    """
    print(f"\n{'=' * 60}")
    print(f"Дедупликация: {os.path.basename(parquet_path)}")
    print(f"{'=' * 60}")
    
    if not os.path.exists(parquet_path):
        print(f"Файл не найден: {parquet_path}")
        return None
    
    # Читаем parquet
    print(f"Читаю parquet файл...")
    df = pl.read_parquet(parquet_path)
    original_count = len(df)
    print(f"   Исходное количество: {original_count:,}")
    
    # Преобразуем даты
    print(f"Обрабатываю даты...")
    df = df.with_columns(
        pl.col("date")
        .str.replace(r"\+00:00$", "")
        .str.replace(r"Z$", "")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S", strict=False)
        .alias("date_dt")
    )
    
    # Сортируем по дате
    df = df.sort("date_dt")
    
    # Проверяем наличие колонки channel
    has_channel = "channel" in df.columns
    if not has_channel:
        # Если нет колонки channel, создаем пустую или берем из имени файла
        channel_name = os.path.basename(parquet_path).replace("embeddings_", "").replace(".parquet", "")
        df = df.with_columns(pl.lit(channel_name).alias("channel"))
    
    # Батчевая обработка с перекрытием для сохранения дубликатов на границах
    N = len(df)
    window = np.timedelta64(WINDOW_HOURS, "h")
    
    # Размер батча (обрабатываем по 500k записей за раз)
    BATCH_SIZE = 500000
    # Перекрытие = размер окна, чтобы не потерять дубликаты на границах
    OVERLAP = WINDOW_HOURS  # часов перекрытия
    
    # Список дубликатов с полной информацией
    duplicates_data = []
    skipped_idx = set()
    
    print(f"Запускаю дедупликацию (батчами)...")
    print(f"   Временное окно: {WINDOW_HOURS} часов")
    print(f"   Порог похожести: {COSINE_THRESHOLD}")
    print(f"   Размер батча: {BATCH_SIZE:,} записей")
    print(f"   Перекрытие: {OVERLAP} часов (для сохранения дубликатов на границах)")
    
    # Получаем даты для определения границ батчей
    dates = df["date_dt"].to_numpy()
    
    # Обрабатываем батчами
    total_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in tqdm(range(total_batches), desc="   Батчи"):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, N)
        
        # Определяем временные границы батча с перекрытием
        batch_start_time = dates[start_idx]
        # Для последнего батча берем до конца, для остальных - с перекрытием
        if batch_num < total_batches - 1:
            batch_end_time = dates[end_idx - 1] + window
            # Находим реальный end_idx с учетом перекрытия
            actual_end_idx = min(N, start_idx + BATCH_SIZE)
            # Расширяем до конца окна
            while actual_end_idx < N and dates[actual_end_idx] <= batch_end_time:
                actual_end_idx += 1
        else:
            actual_end_idx = N
        
        # Извлекаем данные для батча
        batch_df = df[start_idx:actual_end_idx]
        batch_dates = dates[start_idx:actual_end_idx]
        batch_msg_ids = batch_df["message_id"].to_list()
        batch_channels = batch_df["channel"].to_list()
        batch_embeddings_list = batch_df["embedding"].to_list()
        batch_original_dates = batch_df["date"].to_list()
        
        # Конвертируем эмбеддинги батча в numpy матрицу
        batch_emb_matrix = np.stack([
            np.array(emb, dtype=np.float32)
            for emb in batch_embeddings_list
        ])
        
        batch_N = len(batch_df)
        
        # Дедупликация внутри батча
        # Но учитываем, что записи из предыдущего батча уже могли быть обработаны
        for i in tqdm(range(batch_N), desc=f"      Батч {batch_num+1}/{total_batches}", leave=False):
            global_i = start_idx + i
            
            # Пропускаем, если уже помечен как дубликат
            if global_i in skipped_idx:
                continue
            
            j = i + 1
            while j < batch_N and (batch_dates[j] - batch_dates[i]) <= window:
                global_j = start_idx + j
                
                # Пропускаем, если уже помечен как дубликат
                if global_j in skipped_idx:
                    j += 1
                    continue
                
                sim = cosine_sim(batch_emb_matrix[i], batch_emb_matrix[j])
                if sim >= COSINE_THRESHOLD:
                    # j - дубликат i, сохраняем информацию о дубликате
                    dt = batch_dates[j] - batch_dates[i]
                    dt_hours = dt / np.timedelta64(1, 'h')
                    
                    # Используем данные из исходного DataFrame, а не из батча
                    # embedding берем из batch_embeddings_list, который уже список (как в основном файле)
                    duplicates_data.append({
                        "channel": df["channel"][global_j],
                        "message_id": int(df["message_id"][global_j]),
                        "date": df["date"][global_j],
                        "embedding": batch_embeddings_list[j],  # Используем список из батча
                        "base_channel": df["channel"][global_i],
                        "base_message_id": int(df["message_id"][global_i]),
                        "sim": float(sim),
                        "dt": float(dt_hours)
                    })
                    skipped_idx.add(global_j)
                j += 1
        
        # Освобождаем память
        del batch_emb_matrix, batch_df, batch_embeddings_list
        import gc
        gc.collect()
    
    print(f"\nРезультаты:")
    print(f"   Исходное количество: {original_count:,}")
    print(f"   Найдено дубликатов: {len(duplicates_data):,}")
    
    if duplicates_data:
        # Индексы дубликатов
        duplicate_indices = set(skipped_idx)
        indices_to_keep = [i for i in range(N) if i not in duplicate_indices]
        
        # Уникальные записи
        df_unique = df[indices_to_keep]
        df_unique = df_unique.drop("date_dt")
        
        # Дубликаты
        df_duplicates = pl.DataFrame(duplicates_data)
        
        print(f"   Уникальных записей: {len(df_unique):,}")
        print(f"   Дубликатов: {len(df_duplicates):,}")
        
        # Сохраняем уникальные записи
        if output_path is None:
            output_path = parquet_path
        
        print(f"\nСохраняю уникальные записи в: {output_path}")
        df_unique.write_parquet(output_path, compression="snappy")
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"   Размер файла: {size_mb:.2f} MB")
        
        # Сохраняем дубликаты
        if duplicates_path is None:
            # Создаем имя файла для дубликатов
            base_name = os.path.splitext(os.path.basename(parquet_path))[0]
            dir_name = os.path.dirname(parquet_path) or "."
            duplicates_path = os.path.join(dir_name, f"{base_name}_duplicates.parquet")
        
        print(f"Сохраняю дубликаты в: {duplicates_path}")
        df_duplicates.write_parquet(duplicates_path, compression="snappy")
        dup_size_mb = os.path.getsize(duplicates_path) / (1024**2)
        print(f"   Размер файла: {dup_size_mb:.2f} MB")
        
        return df_unique, df_duplicates
    else:
        print(f"   Дубликатов не найдено")
        df = df.drop("date_dt")
        return df, None


def deduplicate_all_parquet_files(embeddings_dir="/kaggle/working"):
    """Дедупликация всех parquet файлов в директории"""
    parquet_files = sorted([
        f for f in os.listdir(embeddings_dir) 
        if f.startswith("embeddings_") and f.endswith(".parquet")
    ])
    
    if not parquet_files:
        print(f"Parquet файлы не найдены в {embeddings_dir}")
        return
    
    print(f"Найдено {len(parquet_files)} parquet файлов")
    
    for f in parquet_files:
        file_path = os.path.join(embeddings_dir, f)
        deduplicate_parquet_file(file_path)
    
    print(f"\nДедупликация всех файлов завершена")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Всегда обрабатываем all_embeddings.parquet в директории скрипта
    script_dir = Path(__file__).parent.absolute()
    default_file = script_dir / "all_embeddings.parquet"
    
    if default_file.exists():
        print(f"Обрабатываю файл: {default_file}")
        deduplicate_parquet_file(str(default_file))
    else:
        print(f"Файл {default_file} не найден")
        print("Поместите all_embeddings.parquet в каталог со скриптом")
