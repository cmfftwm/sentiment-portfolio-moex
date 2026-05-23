import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
import numpy as np
import sqlite3
import os
import shutil
import sys
import warnings
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import gc

warnings.simplefilter('ignore')

# Giga-Embeddings-instruct - публичная модель от Sber AI, токен не требуется
# Модель использует Latent-Attention pooling вместо mean pooling
# Размерность эмбеддингов: 2048 
# Путь к базе данных (только для чтения исходных сообщений)
DB_INPUT_PATH = "/kaggle/input/telegram-messages/telegram_messages.db"

# Для Kaggle: если БД в input, используем напрямую (только чтение)
if "/kaggle/input" in DB_INPUT_PATH:
    DB_PATH = DB_INPUT_PATH  # Только чтение, не копируем
else:
    DB_PATH = DB_INPUT_PATH

# Подключаемся к БД (только для чтения)
conn = sqlite3.connect(DB_PATH)

# ============================================
# НАСТРОЙКА: Укажите каналы для обработки
# ============================================
# Если список пустой [] - обрабатываются ВСЕ каналы
# Если указаны каналы - обрабатываются ТОЛЬКО указанные

# МАЛЕНЬКИЕ КАНАЛЫ (15 каналов для быстрого тестирования)
CHANNELS_SMALL = [
    'messages_headlines_quants',      # 1,210
    'messages_headlines_geo',          # 2,452
    'messages_Dividend_News100',      # 4,191
    'messages_headlines_MACRO',       # 4,372
    'messages_if_stocks',             # 6,792
    'messages_econs',                # 7,658
    'messages_investfuture',          # 9,053
    'messages_sosisochniyparserru',  # 13,258
    'messages_FatCat18',              # 14,744
    'messages_AK47pfl',                # 14,970
    'messages_frank_media',            # 17,702
    'messages_economika',              # 21,986
    'messages_bankser',                # 23,369
    'messages_investingcorp',          # 25,266
    'messages_BIoomberg',              # 37,811
]

# БОЛЬШИЕ КАНАЛЫ (14 каналов)
CHANNELS_LARGE = [
    'messages_vedomosti',             # 56,306
    'messages_Stock_News100',           # 62,105
    'messages_headlines_for_traders',   # 62,341
    'messages_banksta',                 # 64,489
    'messages_cbrstocks',               # 68,787
    'messages_if_market_news',          # 70,042
    'messages_forbesrussia',            # 71,548
    'messages_kommersant',              # 76,340
    'messages_newssmartlab',            # 102,558
    'messages_rbc_news',                # 116,917
    'messages_rt_russian',             # 209,529
    'messages_markettwits',            # 276,665
    'messages_rian_ru',                # 280,501
    'messages_tass_agency',            # 321,883
]

# Выберите группу для обработки:
# CHANNELS_TO_PROCESS = CHANNELS_SMALL  # Для тестирования (15 маленьких каналов)
# CHANNELS_TO_PROCESS = CHANNELS_LARGE  # Для больших каналов
# CHANNELS_TO_PROCESS = []              # Все каналы
CHANNELS_TO_PROCESS = CHANNELS_SMALL  # По умолчанию: маленькие для тестирования

# ============================================

# Список таблиц-каналов (исключаем служебные таблицы)
cursor = conn.cursor()
cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'message_embeddings';"
)
all_tables = [row[0] for row in cursor.fetchall()]

# Фильтруем таблицы по списку каналов
if CHANNELS_TO_PROCESS:
    # Проверяем, что все указанные каналы существуют
    missing_channels = set(CHANNELS_TO_PROCESS) - set(all_tables)
    if missing_channels:
        print(f"ВНИМАНИЕ: Указанные каналы не найдены в БД:")
        for ch in missing_channels:
            print(f"   - {ch}")
        print()
    
    # Фильтруем только существующие каналы
    tables = [t for t in CHANNELS_TO_PROCESS if t in all_tables]
    print(f"Режим: обработка указанных каналов")
    print(f"   Указано каналов: {len(CHANNELS_TO_PROCESS)}")
    print(f"   Найдено в БД: {len(tables)}")
    if len(tables) != len(CHANNELS_TO_PROCESS):
        print(f"   Пропущено (не найдено): {len(CHANNELS_TO_PROCESS) - len(tables)}")
else:
    tables = all_tables
    print(f"Режим: обработка всех каналов")

print(f"Каналы для обработки ({len(tables)}):")
for i, table in enumerate(tables, 1):
    print(f"   {i}. {table}")
print()

# Директория для сохранения parquet файлов
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working"
else:
    OUTPUT_DIR = "./embeddings"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Директория для эмбеддингов: {OUTPUT_DIR}")

MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
MAX_LENGTH = 4096  # Giga-Embeddings-instruct поддерживает до 4096 токенов

# Giga-Embeddings-instruct использует Latent-Attention pooling
# Эмбеддинги возвращаются напрямую через return_embeddings=True
# Функция mean_pooling не нужна для этой модели

def encode(texts, models, tokenizer, devices, batch_size=4):
    """Параллельная обработка на нескольких GPU с правильной синхронизацией"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна! Убедитесь, что GPU включены в Kaggle.")

    num_devices = len(devices)
    streams = [torch.cuda.Stream(device=dev) for dev in devices]

    batches = []
    for batch_id, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batches.append((batch_id, batch))

    total_batches = len(batches)

    all_embeddings = []
    next_batch = 0

    with tqdm(total=total_batches) as pbar:
        gpu_busy = [None] * num_devices
        while len(all_embeddings) < total_batches:
            
            # Проверяем завершенные задачи на GPU
            for gpu_idx in range(num_devices):
                if gpu_busy[gpu_idx] is None:
                    continue
                    
                batch_id = gpu_busy[gpu_idx]['batch_id']
                stream = streams[gpu_idx]

                # Проверяем, завершен ли поток (без блокирующей синхронизации)
                if stream.query():
                    # Поток завершен, синхронизируем для безопасного доступа к результатам
                    stream.synchronize()
                    emb = gpu_busy[gpu_idx]['emb']
                    # Убеждаемся, что эмбеддинг на GPU перед переносом на CPU
                    if emb.is_cuda:
                        all_embeddings.append((batch_id, emb.cpu()))
                    else:
                        all_embeddings.append((batch_id, emb))
                    gpu_busy[gpu_idx] = None
                    pbar.update(1)

            # Запускаем новые задачи на свободных GPU
            for gpu_idx in range(num_devices):

                if next_batch >= total_batches:
                    continue  # batch-и закончились

                if gpu_busy[gpu_idx] is not None:
                    continue  # GPU занята

                batch_id, batch = batches[next_batch]
                next_batch += 1

                dev = devices[gpu_idx]
                model = models[gpu_idx]
                stream = streams[gpu_idx]

                # Убеждаемся, что модель на правильном устройстве
                if next(model.parameters()).device != torch.device(dev):
                    model = model.to(dev)

                with torch.cuda.stream(stream):
                    inputs = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=MAX_LENGTH
                    )
                    
                    # Переносим на GPU с использованием pin_memory для ускорения
                    inputs = {k: v.pin_memory().to(dev, non_blocking=True) for k, v in inputs.items()}
                    
                    with torch.inference_mode():
                        # Giga-Embeddings-instruct возвращает эмбеддинги напрямую через return_embeddings=True
                        # Модель использует Latent-Attention pooling вместо mean pooling
                        emb = model(**inputs, return_embeddings=True)
                        # Эмбеддинги уже нормализованы моделью
                        
                gpu_busy[gpu_idx] = {"batch_id": batch_id, "emb": emb}

    # Синхронизируем все потоки и собираем оставшиеся результаты
    for gpu_idx in range(num_devices):
        if gpu_busy[gpu_idx] is not None:
            streams[gpu_idx].synchronize()  # Синхронизируем конкретный поток
            batch_id = gpu_busy[gpu_idx]['batch_id']
            emb = gpu_busy[gpu_idx]['emb']
            if emb.is_cuda:
                all_embeddings.append((batch_id, emb.cpu()))
            else:
                all_embeddings.append((batch_id, emb))

    all_embeddings.sort(key=lambda x: x[0])
    all_embeddings = [emb[1] for emb in all_embeddings]

    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Конвертируем bfloat16/float16 в float32 для совместимости с numpy
    if embeddings.dtype in (torch.float16, torch.bfloat16):
        embeddings = embeddings.float()

    return embeddings.cpu().numpy()

# Проверяем доступность CUDA и определяем устройства
print(f"\nПроверка доступных GPU...")
print(f"   CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"   Количество GPU: {num_gpus}")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    print(f"   Используем {num_gpus} GPU: {devices}")
else:
    print("   CUDA недоступна! Убедитесь, что GPU включены в Kaggle (Settings -> Accelerator -> GPU T4 x2)")
    sys.exit(1)
sys.stdout.flush()

print(f"\nЗагрузка модели {MODEL_NAME}...")
print(f"   Модель: Giga-Embeddings-instruct (Sber AI)")
print(f"   Размерность эмбеддингов: 2048")
print(f"   Максимальная длина: {MAX_LENGTH} токенов")
print(f"   Pooling: Latent-Attention (встроенный)")
sys.stdout.flush()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

models = []
for i, dev in enumerate(devices):
    print(f"   Загрузка модели на {dev}...")
    sys.stdout.flush()
    # Giga-Embeddings-instruct использует bfloat16 и flash_attention_2 для ускорения
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    except Exception as e:
        # Если flash_attention_2 недоступен, используем стандартную реализацию
        print(f"   flash_attention_2 недоступен: {e}, используем стандартную реализацию")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    model.to(dev)
    model.eval()
    
    # Компиляция модели для ускорения (PyTorch 2.0+)
    try:
        model.compile()
        print(f"   Модель скомпилирована на {dev}")
    except Exception as e:
        print(f"   Компиляция недоступна на {dev}: {e}")
    
    # Проверяем, что модель действительно на GPU
    device_check = next(model.parameters()).device
    if device_check != torch.device(dev):
        print(f"   ВНИМАНИЕ: Модель на {device_check}, ожидалось {dev}")
    else:
        print(f"   Модель загружена на {dev}")
    
    models.append(model)
    sys.stdout.flush()

print(f"Все модели загружены и готовы к работе\n")
sys.stdout.flush()

# Проходим по всем таблицам
total_channels = len(tables)
print(f"\n{'=' * 60}")
print(f"Начинаю обработку {total_channels} каналов")
print(f"{'=' * 60}\n")

for channel_num, table in enumerate(tables, 1):
    print(f"\n{'=' * 60}")
    print(f"Обработка таблицы: {table} ({channel_num}/{total_channels})")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    # Читаем message_id + message + date
    print("   Чтение сообщений из БД...")
    sys.stdout.flush()
    query = f"SELECT message_id, message, date FROM {table}"
    df = pl.read_database(query, conn)

    # Выкидываем пустые сообщения (NULL, пустые строки и только пробелы)
    df = df.filter(
        pl.col("message").is_not_null() & 
        (pl.col("message").str.strip_chars() != "")
    )
    print(f"   Найдено сообщений в таблице (непустых): {len(df)}")
    sys.stdout.flush()

    # Проверим, сколько уже есть эмбеддингов для этого канала (читаем из parquet)
    print("   Проверка уже обработанных сообщений...")
    sys.stdout.flush()
    parquet_path = os.path.join(OUTPUT_DIR, f"embeddings_{table}.parquet")
    existing_ids_set = set()
    
    if os.path.exists(parquet_path):
        try:
            existing_df = pl.read_parquet(parquet_path)
            existing_ids_set = set(existing_df["message_id"].to_list())
            print(f"   Уже обработано: {len(existing_ids_set)} (из файла {parquet_path})")
        except Exception as e:
            print(f"   Ошибка при чтении существующего файла: {e}")
            existing_ids_set = set()
    else:
        print(f"   Файл не существует, начинаем с нуля")
    
    sys.stdout.flush()

    # Фильтруем уже обработанные сообщения
    df = df.filter(~pl.col("message_id").is_in(existing_ids_set))

    if len(df) == 0:
        print("   Всё уже посчитано, пропускаю.")
        sys.stdout.flush()
        continue

    print(f"   Сообщений к обработке: {len(df)}")
    sys.stdout.flush()

    # Батчами считаем эмбеддинги
    processed_count = 0
    total_rows = len(df)
    batch_size = 64  # Размер батча для обработки
    encode_batch_size = 8  # Размер батча для encode
    SAVE_INTERVAL = 5  # Сохраняем каждые 5 батчей
    
    # Список для накопления данных перед сохранением
    all_data = []
    
    # Общий прогресс-бар для всех сообщений
    pbar = tqdm(total=total_rows, desc="   Обработка сообщений", unit="msg", unit_scale=True)

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = df.slice(start, end - start)

        texts = batch["message"].to_list()
        ids = batch["message_id"].to_list()
        dates = batch["date"].to_list()

        # Получаем эмбеддинги
        emb_array = encode(texts, models, tokenizer, devices, batch_size=encode_batch_size)

        # Сохраняем данные в список (эмбеддинги конвертируем в списки для parquet)
        for msg_id, date, emb in zip(ids, dates, emb_array):
            all_data.append({
                "message_id": msg_id,
                "date": date,
                "embedding": emb.tolist(),  # конвертируем numpy array в список
                "sentiment": None
            })
        
        processed_count += len(ids)
        batch_num = len(all_data) // batch_size
        
        # Обновляем общий прогресс-бар
        pbar.update(len(ids))
        
        # Сохраняем периодически для экономии памяти
        if len(all_data) >= batch_size * SAVE_INTERVAL:
            # Создаем DataFrame из накопленных данных
            new_df = pl.DataFrame({
                "message_id": [d["message_id"] for d in all_data],
                "date": [d["date"] for d in all_data],
                "embedding": [d["embedding"] for d in all_data],
                "sentiment": [d["sentiment"] for d in all_data]
            })
            
            # Читаем существующий файл, если есть
            if os.path.exists(parquet_path):
                try:
                    existing_df = pl.read_parquet(parquet_path)
                    # Объединяем, исключая дубликаты по message_id
                    combined_df = pl.concat([existing_df, new_df]).unique(subset=["message_id"], keep="last")
                except Exception as e:
                    print(f"   Ошибка при чтении существующего файла, создаем новый: {e}")
                    combined_df = new_df
            else:
                combined_df = new_df
            
            # Сортируем по дате перед сохранением
            combined_df = combined_df.sort("date")
            
            # Сохраняем в parquet с сжатием
            combined_df.write_parquet(parquet_path, compression="snappy")
            all_data = []  # Очищаем список после сохранения

        # Очистка памяти
        del emb_array

        # Очистка памяти GPU между батчами
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Закрываем прогресс-бар
    pbar.close()
    
    # Сохраняем оставшиеся данные
    if all_data:
        new_df = pl.DataFrame({
            "message_id": [d["message_id"] for d in all_data],
            "date": [d["date"] for d in all_data],
            "embedding": [d["embedding"] for d in all_data],
            "sentiment": [d["sentiment"] for d in all_data]
        })
        
        if os.path.exists(parquet_path):
            try:
                existing_df = pl.read_parquet(parquet_path)
                combined_df = pl.concat([existing_df, new_df]).unique(subset=["message_id"], keep="last")
            except Exception:
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Сортируем по дате перед сохранением
        combined_df = combined_df.sort("date")
        
        combined_df.write_parquet(parquet_path, compression="snappy")
    
    gc.collect()
    # Финальная очистка GPU памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"   Обработано сообщений в таблице {table}: {processed_count}")
    print(f"   Сохранено в: {parquet_path}")
    sys.stdout.flush()

print(f"\n{'=' * 60}")
print(f"Готово! Обработано {total_channels} каналов.")
print("Все эмбеддинги записаны в parquet файлы.")
print(f"{'=' * 60}\n")

# Для Kaggle: файлы в /kaggle/working, можно скачать из Output
if os.path.exists("/kaggle/working"):
    print(f"Эмбеддинги сохранены в директории: {OUTPUT_DIR}")
    print(f"   Файлы (по одному на канал):")
    parquet_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.parquet')])
    for f in parquet_files:
        file_path = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   - {f} ({size_mb:.1f} MB)")
    print(f"\n   Вы можете скачать файлы из Kaggle Output после завершения работы")
    print(f"   Каждый файл можно скачать отдельно!")

conn.close()