# Установка необходимых пакетов для Kaggle
import subprocess
import sys

def install_package(package):
    """Устанавливает пакет, если он не установлен"""
    try:
        __import__(package)
    except ImportError:
        print(f"Устанавливаю {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"{package} установлен")

# Устанавливаем зависимости для geracl
install_package("loguru")
install_package("pytorch-lightning")

try:
    import geracl
except ImportError:
    print("Устанавливаю geracl из GitHub...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/deepvk/GeRaCl.git", "-q"
    ])
    print("geracl установлен")

import sqlite3
import polars as pl
import numpy as np
import torch
import json
import re
import os
import sys
import gc
from tqdm import tqdm
from transformers import AutoTokenizer
from geracl import GeraclHF
from geracl.data.batch_creation import prepare_inference_batch

# Путь к базе данных (для Kaggle)
DB_INPUT_PATH = "/kaggle/input/datasets/markbabii/telegram-messages-new/telegram_messages_new.db"
DB_LOCAL_PATH = "telegram_messages_new.db"

# Для Kaggle: если БД в input, используем напрямую (только чтение)
if os.path.exists(DB_INPUT_PATH):
    DB_PATH = DB_INPUT_PATH
    print(f"Найдена БД на Kaggle: {DB_INPUT_PATH}")
elif os.path.exists(DB_LOCAL_PATH):
    DB_PATH = DB_LOCAL_PATH
    print(f"Найдена локальная БД: {DB_LOCAL_PATH}")
else:
    print(f"БД не найдена: {DB_INPUT_PATH}")
    raise SystemExit(1)

MODEL_NAME = "deepvk/GeRaCl-USER2-base"

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
    device = torch.device(devices[0])
    device_str = devices[0]
else:
    print("   CUDA недоступна, используется CPU")
    devices = ["cpu"]
device = torch.device("cpu")
device_str = "cpu"
sys.stdout.flush()

print(f"\nЗагрузка модели {MODEL_NAME}...")
sys.stdout.flush()

import copy
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"   Загрузка базовой модели...")
sys.stdout.flush()
base_model = GeraclHF.from_pretrained(MODEL_NAME)
# Отключаем torch.compile внутри ModernBERT — вызывает device mismatch при multi-GPU
if hasattr(base_model, '_classification_core') and hasattr(base_model._classification_core, '_token_embedder'):
    base_model._classification_core._token_embedder.config.reference_compile = False

models = []
for i, dev in enumerate(devices):
    model = base_model if i == 0 else copy.deepcopy(base_model)
    model.config.device = dev
    model = model.to(dev).eval()
    if hasattr(model, '_classification_core'):
        model._classification_core._device = dev
    models.append(model)
    print(f"   Модель загружена на {dev}")
    sys.stdout.flush()

del base_model
gc.collect()
print(f"Все модели загружены и готовы к работе\n")
sys.stdout.flush()

label_space = ["негативный", "нейтральный", "позитивный"]


# ---------- Функция параллельной обработки на нескольких GPU ----------
def process_sentiment_batch_parallel(batch_texts, labels, models, tokenizer, devices, batch_size=4):
    """Параллельная обработка сентимента на нескольких GPU с правильной синхронизацией"""
    if not torch.cuda.is_available() or len(devices) == 1:
        # Если одна GPU или CPU, используем простую обработку
        input_ids, attention_mask, classes_mask, classes_count = prepare_inference_batch(
            batch_texts, [labels] * len(batch_texts), tokenizer
        )
        input_ids = input_ids.to(devices[0])
        attention_mask = attention_mask.to(devices[0])
        classes_mask = classes_mask.to(devices[0])
        classes_count = classes_count.to(devices[0])
        
        with torch.no_grad():
            similarities = models[0](input_ids, attention_mask, classes_mask, classes_count)
        
        return similarities, input_ids, attention_mask, classes_mask, classes_count
    
    num_devices = len(devices)
    streams = [torch.cuda.Stream(device=dev) for dev in devices]
    
    batches = []
    for batch_id, i in enumerate(range(0, len(batch_texts), batch_size)):
        batch = batch_texts[i:i + batch_size]
        batches.append((batch_id, batch))
    
    total_batches = len(batches)
    all_similarities = []
    all_inputs = []
    next_batch = 0
    
    gpu_busy = [None] * num_devices
    
    while len(all_similarities) < total_batches:
        # Проверяем завершенные задачи на GPU
        for gpu_idx in range(num_devices):
            if gpu_busy[gpu_idx] is None:
                continue
            
            batch_id = gpu_busy[gpu_idx]['batch_id']
            stream = streams[gpu_idx]
            
            if stream.query():
                stream.synchronize()
                result = gpu_busy[gpu_idx]['result']
                inputs = gpu_busy[gpu_idx]['inputs']
                all_similarities.append((batch_id, result))
                all_inputs.append((batch_id, inputs))
                gpu_busy[gpu_idx] = None
        
        # Запускаем новые задачи на свободных GPU
        for gpu_idx in range(num_devices):
            if next_batch >= total_batches:
                continue
            
            if gpu_busy[gpu_idx] is not None:
                continue
            
            batch_id, batch = batches[next_batch]
            next_batch += 1
            
            dev = devices[gpu_idx]
            model = models[gpu_idx]
            stream = streams[gpu_idx]
            
            with torch.cuda.stream(stream):
                input_ids, attention_mask, classes_mask, classes_count = prepare_inference_batch(
                    batch, [labels] * len(batch), tokenizer
                )
                input_ids = input_ids.to(dev, non_blocking=True)
                attention_mask = attention_mask.to(dev, non_blocking=True)
                classes_mask = classes_mask.to(dev, non_blocking=True)
                classes_count = classes_count.to(dev, non_blocking=True)
                
                with torch.no_grad():
                    similarities = model(input_ids, attention_mask, classes_mask, classes_count)
                
                gpu_busy[gpu_idx] = {
                    "batch_id": batch_id,
                    "result": similarities,
                    "inputs": (input_ids, attention_mask, classes_mask, classes_count)
                }
    
    # Синхронизируем все потоки
    for gpu_idx in range(num_devices):
        if gpu_busy[gpu_idx] is not None:
            streams[gpu_idx].synchronize()
            batch_id = gpu_busy[gpu_idx]['batch_id']
            result = gpu_busy[gpu_idx]['result']
            inputs = gpu_busy[gpu_idx]['inputs']
            all_similarities.append((batch_id, result))
            all_inputs.append((batch_id, inputs))
    
    # Сортируем и объединяем результаты
    all_similarities.sort(key=lambda x: x[0])
    all_inputs.sort(key=lambda x: x[0])
    
    # Объединяем similarities (переносим на CPU перед объединением)
    similarities_list = [sim[1].cpu() for sim in all_similarities]
    if len(similarities_list) > 1:
        similarities = torch.cat(similarities_list, dim=0)
    else:
        similarities = similarities_list[0]
    
    # Объединяем classes_count из всех батчей
    classes_count_list = []
    for inputs_tuple in all_inputs:
        _, _, _, classes_count_batch = inputs_tuple[1]
        classes_count_list.append(classes_count_batch.cpu())
    
    if len(classes_count_list) > 1:
        classes_count = torch.cat(classes_count_list, dim=0)
    else:
        classes_count = classes_count_list[0]
    
    # Возвращаем первый набор inputs для совместимости (не используются дальше)
    input_ids, attention_mask, classes_mask, _ = all_inputs[0][1]
    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    classes_mask = classes_mask.cpu()
    
    return similarities, input_ids, attention_mask, classes_mask, classes_count


# ---------- Функция извлечения контекста вокруг тикера ----------
def extract_ticker_context(text, ticker, context_sentences=1):
    """
    Извлекает контекст вокруг упоминания тикера в тексте.
    
    Args:
        text: полный текст новости
        ticker: тикер для поиска
        context_sentences: количество соседних предложений с каждой стороны (по умолчанию 1)
    
    Returns:
        Строка с контекстом вокруг тикера, или весь текст, если тикер не найден
    """
    if not text or not ticker:
        return text or ""
    
    # Разбиваем текст на предложения
    # Используем более точное разбиение: точка/восклицательный/вопросительный знак + пробел или конец строки
    sentence_pattern = r'([.!?]+(?:\s+|$))'
    parts = re.split(sentence_pattern, text)
    
    # Объединяем части в предложения
    sentences = []
    for i in range(0, len(parts), 2):
        if i < len(parts):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]  # Добавляем разделитель
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    
    if not sentences:
        return text
    
    # Ищем предложения, содержащие тикер (регистронезависимо)
    # Используем границы слов для точного поиска
    ticker_pattern = rf'\b{re.escape(ticker)}\b'
    
    relevant_indices = []
    for i, sentence in enumerate(sentences):
        if re.search(ticker_pattern, sentence, re.IGNORECASE):
            relevant_indices.append(i)
    
    if not relevant_indices:
        # Если тикер не найден, возвращаем весь текст
        return text
    
    # Собираем контекст: предложения с тикером + соседние
    context_indices = set()
    for idx in relevant_indices:
        # Добавляем само предложение
        context_indices.add(idx)
        # Добавляем соседние предложения
        for offset in range(1, context_sentences + 1):
            if idx - offset >= 0:
                context_indices.add(idx - offset)
            if idx + offset < len(sentences):
                context_indices.add(idx + offset)
    
    # Сортируем индексы и собираем контекст
    context_sentences_list = [sentences[i] for i in sorted(context_indices)]
    context = ' '.join(context_sentences_list).strip()
    
    # Если контекст слишком короткий (меньше 30 символов), возвращаем весь текст
    if len(context) < 30:
        return text
    
    return context


# ---------- Функция обработки и сохранения сентимента по тикерам ----------
def process_and_save_ticker_sentiment(texts, labels, channels_list, msg_ids_list, tickers_list, output_path, models, tokenizer, devices, batch_size=64, gpu_batch_size=4):
    """
    Обрабатывает тексты батчами и сохраняет результаты сентимента для каждого тикера в parquet файл.
    Для каждого тикера извлекается контекст вокруг его упоминания и анализируется отдельно.
    
    texts: список строк (полные тексты новостей)
    labels: ["негативный", "нейтральный", "позитивный"]
    channels_list: список каналов
    msg_ids_list: список message_id
    tickers_list: список списков тикеров для каждой новости (может быть пустым)
    output_path: путь к parquet файлу для сохранения результатов
    """
    COMMIT_INTERVAL = 10  # Сохраняем каждые 10 батчей
    
    # Собираем статистику для вывода
    all_sentiments = []
    total_ticker_records = 0
    
    # Список для накопления данных перед сохранением
    all_results = []
    
    # Подготавливаем данные: для каждой новости создаем отдельные тексты для каждого тикера
    ticker_texts = []
    ticker_metadata = []  # (channel, message_id, ticker, date)
    
    print("Извлекаю контекст для каждого тикера...")
    dates_list = df["date"].to_list() if "date" in df.columns else [None] * len(texts)
    
    for text, channel, msg_id, tickers, date in zip(texts, channels_list, msg_ids_list, tickers_list, dates_list):
        if tickers:  # Если есть тикеры
            for ticker in tickers:
                # Извлекаем контекст вокруг тикера
                context = extract_ticker_context(text, ticker, context_sentences=1)
                ticker_texts.append(context)
                ticker_metadata.append((channel, msg_id, ticker, date))
        # Если тикеров нет, пропускаем эту новость
    
    if not ticker_texts:
        print("Нет новостей с тикерами для обработки")
        return
    
    print(f"Всего контекстов для анализа: {len(ticker_texts):,}")
    
    # Обрабатываем батчами
    total_batches = (len(ticker_texts) + batch_size - 1) // batch_size
    batch_num = 0

    for i in tqdm(range(0, len(ticker_texts), batch_size), desc="Анализ сентимента по тикерам", unit="батч", total=total_batches):
        batch_texts = ticker_texts[i:i + batch_size]
        batch_metadata = ticker_metadata[i:i + batch_size]

        with torch.no_grad():
            # Используем параллельную обработку на нескольких GPU
            similarities, input_ids, attention_mask, classes_mask, classes_count = process_sentiment_batch_parallel(
                batch_texts, labels, models, tokenizer, devices, batch_size=gpu_batch_size
            )
            
            # Преобразуем similarities в вероятности для каждого текста в батче
            batch_probs = []
            start_idx = 0
            for count in classes_count:
                end_idx = start_idx + count.item()
                text_similarities = similarities[start_idx:end_idx]
                text_probs = torch.softmax(text_similarities, dim=0)
                batch_probs.append(text_probs)
                start_idx = end_idx
            
            batch_probs = torch.stack(batch_probs, dim=0).cpu()
            
            # Очистка памяти GPU
            del similarities, input_ids, attention_mask, classes_mask, classes_count
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Извлекаем вероятности в правильном порядке
            neg = batch_probs[:, 0].numpy()  # негативный (индекс 0)
            neu = batch_probs[:, 1].numpy()  # нейтральный (индекс 1)
            pos = batch_probs[:, 2].numpy()  # позитивный (индекс 2)
            
            # Очистка памяти после извлечения
            del batch_probs
            
            # Проверка: сумма вероятностей должна быть близка к 1.0
            probs_sum = neg + neu + pos
            if not np.allclose(probs_sum, 1.0, atol=0.01):
                print(f"Внимание: сумма вероятностей не равна 1.0! Среднее: {probs_sum.mean():.6f}")
            
            # Преобразуем в шкалу от -1 до 1
            sentiment_scores = np.where(
                neu > 0.5,  # Если нейтральность доминирует
                0.0,  # То sentiment = 0 (нейтральный)
                pos - neg  # Иначе разница между позитивным и негативным
            )
            all_sentiments.extend(sentiment_scores.tolist())
            
            # Подготавливаем данные для сохранения
            for (sent, n, u, p), (channel, msg_id, ticker, date) in zip(
                zip(sentiment_scores, neg, neu, pos), batch_metadata
            ):
                all_results.append({
                    "channel": channel,
                    "message_id": int(msg_id),
                    "ticker": ticker,
                    "date": date,
                    "sentiment": float(sent),
                    "sentiment_neg": float(n),
                    "sentiment_neu": float(u),
                    "sentiment_pos": float(p)
                })
                total_ticker_records += 1
            
            batch_num += 1
            
            # Сохраняем периодически для экономии памяти
            if batch_num % COMMIT_INTERVAL == 0 and all_results:
                # Создаем DataFrame из накопленных данных
                new_df = pl.DataFrame(all_results)
                
                # Читаем существующий файл, если есть
                if os.path.exists(output_path):
                    try:
                        existing_df = pl.read_parquet(output_path)
                        # Объединяем, исключая дубликаты
                        combined_df = pl.concat([existing_df, new_df]).unique(
                            subset=["channel", "message_id", "ticker"], 
                            keep="last"
                        )
                    except Exception as e:
                        print(f"Ошибка при чтении существующего файла, создаем новый: {e}")
                        combined_df = new_df
                else:
                    combined_df = new_df
                
                # Сортируем по дате перед сохранением
                if "date" in combined_df.columns:
                    combined_df = combined_df.sort("date")
    
                # Сохраняем в parquet с сжатием
                combined_df.write_parquet(output_path, compression="snappy")
                all_results = []  # Очищаем список после сохранения
                
                # Очистка памяти
                del new_df, combined_df
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Сохраняем оставшиеся данные
    if all_results:
        new_df = pl.DataFrame(all_results)
        
        if os.path.exists(output_path):
            try:
                existing_df = pl.read_parquet(output_path)
                combined_df = pl.concat([existing_df, new_df]).unique(
                    subset=["channel", "message_id", "ticker"], 
                    keep="last"
                )
            except Exception:
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Сортируем по дате перед сохранением
        if "date" in combined_df.columns:
            combined_df = combined_df.sort("date")
        
        combined_df.write_parquet(output_path, compression="snappy")
        
        # Очистка памяти
        del new_df, combined_df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Выводим статистику
    if all_sentiments:
        all_sentiments_array = np.array(all_sentiments)
        print(f"\nДиапазон сентимента: от {all_sentiments_array.min():.3f} до {all_sentiments_array.max():.3f}")
        print(f"Всего записей в news_ticker_sentiment: {total_ticker_records:,}")


# ---------- Загружаем все новости с тикерами из таблиц каналов ----------
print(f"Подключаюсь к БД: {DB_PATH}")
if not os.path.exists(DB_PATH):
    print(f"Файл БД не найден: {DB_PATH}")
    print(f"   Проверьте путь к базе данных")
    exit(1)

conn = sqlite3.connect(DB_PATH)

print("Загружаю сообщения с тикерами из таблиц каналов...")
cursor = conn.cursor()

# Сначала получаем ВСЕ таблицы (включая служебные) для диагностики
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
all_tables_raw = [row[0] for row in cursor.fetchall()]
print(f"Всего таблиц в БД (включая служебные): {len(all_tables_raw)}")
if all_tables_raw:
    print(f"   Все таблицы: {all_tables_raw}")

# Получаем все таблицы каналов (исключая служебные)
cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' 
    AND name NOT LIKE 'sqlite_%' 
    AND name != 'news_ticker_sentiment'
""")
all_tables = [row[0] for row in cursor.fetchall()]

print(f"Таблиц каналов (без служебных): {len(all_tables)}")
if all_tables:
    print(f"   Таблицы каналов: {all_tables[:10]}{'...' if len(all_tables) > 10 else ''}")
else:
    print("   В БД нет таблиц каналов!")
    print("   Возможно, БД пустая или используется другая структура")
    print("   Проверьте:")
    print("   1. Правильно ли подключен датасет на Kaggle")
    print("   2. Правильное ли название датасета")
    print("   3. Есть ли файл telegram_messages.db в датасете")
    conn.close()
    raise SystemExit(1)

# Фильтруем таблицы, которые имеют колонку tickers
channel_tables = []
for table in all_tables:
    try:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"   {table}: колонки = {columns}")
        if 'tickers' in columns:
            channel_tables.append(table)
            print(f"      Есть колонка tickers")
    except Exception as e:
        print(f"Ошибка при проверке таблицы {table}: {e}")

if not channel_tables:
    print("Не найдено таблиц каналов с колонкой tickers!")
    print(f"   Проверьте, что в БД есть таблицы с колонкой 'tickers'")
    print(f"   Всего таблиц проверено: {len(all_tables)}")
    conn.close()
    raise SystemExit(1)

print(f"Найдено таблиц каналов с тикерами: {len(channel_tables)}")
print(f"   Таблицы: {channel_tables[:10]}{'...' if len(channel_tables) > 10 else ''}")

# Собираем все сообщения с тикерами из всех таблиц
all_messages = []
for table in channel_tables:
    try:
        # Получаем сообщения с тикерами
        cursor.execute(f"""
            SELECT message_id, COALESCE(date, '') as date
            FROM {table} 
            WHERE tickers IS NOT NULL AND tickers != ''
        """)
        rows = cursor.fetchall()
        count = len(rows)
        if count > 0:
            print(f"   {table}: {count} сообщений с тикерами")
        for msg_id, date in rows:
            all_messages.append({
                'channel': table,
                'message_id': msg_id,
                'date': date
            })
    except Exception as e:
        print(f"Ошибка при чтении таблицы {table}: {e}")
        continue

if not all_messages:
    print("Не найдено сообщений с тикерами!")
    conn.close()
    exit(1)

if not all_messages:
    print("Не найдено сообщений с тикерами!")
    print("   Возможные причины:")
    print("   1. В БД нет таблиц с колонкой 'tickers'")
    print("   2. В таблицах нет сообщений с заполненными тикерами")
    print("   3. БД пустая или повреждена")
    conn.close()
    raise SystemExit(1)

# Создаем DataFrame из собранных данных
df = pl.DataFrame(all_messages)
print(f"Найдено сообщений с тикерами: {len(df):,}")

# Проверяем, что DataFrame не пустой и содержит нужные колонки
if len(df) == 0:
    print("DataFrame пустой после создания!")
    conn.close()
    raise SystemExit(1)

if "channel" not in df.columns or "message_id" not in df.columns:
    print(f"Отсутствуют нужные колонки в DataFrame!")
    print(f"   Доступные колонки: {df.columns}")
    print(f"   Ожидаемые колонки: channel, message_id")
    conn.close()
    raise SystemExit(1)

# ---------- Достаём тексты и тикеры из таблиц-каналов ----------
# Читаем все тексты одним запросом на канал (вместо 11k отдельных запросов)
print("Читаю тексты сообщений...")
from collections import defaultdict

channels_col = df["channel"].to_list()
msg_ids_col  = df["message_id"].to_list()

# Группируем message_id по каналу
ids_by_channel = defaultdict(list)
for ch, mid in zip(channels_col, msg_ids_col):
    ids_by_channel[ch].append(mid)

# Читаем все тексты одним запросом на канал
data_by_key = {}  # (channel, message_id) -> (message, tickers_json)
cursor = conn.cursor()
for ch, mids in ids_by_channel.items():
    placeholders = ",".join("?" * len(mids))
    cursor.execute(
        f'SELECT message_id, message, tickers FROM "{ch}" WHERE message_id IN ({placeholders})',
        mids
    )
    for mid, msg, tickers_json in cursor.fetchall():
        data_by_key[(ch, mid)] = (msg, tickers_json)

# Собираем в правильном порядке
texts = []
tickers_list = []
for ch, mid in zip(channels_col, msg_ids_col):
    row = data_by_key.get((ch, mid))
    if row is None:
        texts.append("")
        tickers_list.append([])
    else:
        texts.append(str(row[0]) if row[0] else "")
        tickers_json = row[1]
        if tickers_json:
            try:
                t = json.loads(tickers_json)
                tickers_list.append(t if isinstance(t, list) else [])
            except (json.JSONDecodeError, TypeError):
                tickers_list.append([])
        else:
            tickers_list.append([])

print("Получено текстов:", len(texts))
print(f"Новостей с тикерами: {sum(1 for t in tickers_list if t)}")

# ---------- Подготавливаем путь для сохранения результатов ----------
# Для Kaggle: создаем parquet файл в working директории
if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working"
else:
    OUTPUT_DIR = "."
    os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PARQUET_PATH = os.path.join(OUTPUT_DIR, "news_ticker_sentiment.parquet")
print(f"Результаты будут сохранены в: {OUTPUT_PARQUET_PATH}")
sys.stdout.flush()

# ---------- Обрабатываем и сохраняем сентимент по тикерам ----------
channels_list = df["channel"].to_list()
msg_ids_list = df["message_id"].to_list()

print("Начинаю обработку сентимента для тикеров...")
sys.stdout.flush()
# Уменьшаем размер батча для экономии памяти GPU
# gpu_batch_size - размер батча для каждой GPU (меньше = меньше памяти)
process_and_save_ticker_sentiment(texts, label_space, channels_list, msg_ids_list, tickers_list, OUTPUT_PARQUET_PATH, models, tokenizer, devices, batch_size=64, gpu_batch_size=8)

conn.close()

print("Готово! Сентимент по тикерам записан в parquet файл.")
sys.stdout.flush()

# Для Kaggle: информация о результатах
if os.path.exists(OUTPUT_PARQUET_PATH):
    file_size_mb = os.path.getsize(OUTPUT_PARQUET_PATH) / (1024 * 1024)
    print(f"\nРезультаты сохранены в: {OUTPUT_PARQUET_PATH}")
    print(f"   Размер файла: {file_size_mb:.1f} MB")
    if os.path.exists("/kaggle/working"):
        print(f"   Вы можете скачать файл из Kaggle Output после завершения работы")