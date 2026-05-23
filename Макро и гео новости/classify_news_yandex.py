"""
Классификация новостей по категориям: macro / geo / company / skip
Модель: yandex/YandexGPT-5-Lite-8B-instruct-GGUF (Q4_K_M, ~5GB)

2-GPU режим: каждый GPU грузит модель отдельно, обрабатывает свою половину.
Kaggle: 2x T4 (14.56GB each) → ~20 it/s, ~11 часов на 800k сообщений.
"""

import os
import subprocess
import sys

# llama-cpp-python с CUDA для GGUF моделей
try:
    from llama_cpp import Llama
except ImportError:
    print("Устанавливаю llama-cpp-python...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124",
        "-q"
    ])
    from llama_cpp import Llama

import sqlite3
import re
import glob as _glob
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# ─── Настройки ──────────────────────────────────────────────────────────────

# Пути к БД (перебираем возможные варианты)
_DB_CANDIDATES = [
    "/kaggle/input/telegram-messages-2/telegram_messages.db",
    "/kaggle/input/telegram-messages/telegram_messages.db",
    "/kaggle/input/telegram-messages_2/telegram_messages.db",
    "/kaggle/input/telegramdb/telegram_messages.db",
    "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages.db",
    "telegram_messages.db",
]
DB_PATH = next((p for p in _DB_CANDIDATES if os.path.exists(p)), None)
if DB_PATH is None:
    _found = _glob.glob("/kaggle/input/**/*.db", recursive=True)
    if _found:
        DB_PATH = _found[0]
        print(f"Найдена БД: {DB_PATH}")
    else:
        raise FileNotFoundError(
            f"telegram_messages.db не найдена.\nПробовали: {_DB_CANDIDATES}"
        )

# Выходные файлы
if os.path.exists("/kaggle/working"):
    WORK_DIR        = "/kaggle/working"
else:
    WORK_DIR        = "."

OUTPUT          = os.path.join(WORK_DIR, "classified_news_yandex.csv")
OUTPUT_RELEVANT = os.path.join(WORK_DIR, "classified_relevant_yandex.csv")

LIMIT            = None   # None = все; число = последние N на канал
CHECKPOINT_EVERY = 500    # сохранять каждые N классификаций

# GGUF модель — Q4_K_M ~5GB, помещается на один T4
MODEL_REPO     = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF"
MODEL_FILENAME = "YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"

print(f"БД: {DB_PATH}")

# ─── Промпт ──────────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """Определи категорию новости для анализа российского фондового рынка (MOEX). Ответь ТОЛЬКО одним словом из списка: macro, geo, company, skip. Никаких объяснений.

Категории:
- macro: ключевая ставка ЦБ, инфляция, ВВП, бюджет, курс рубля, цены на нефть как экономический индикатор
- geo: война на Украине, международные санкции, военные действия, переговоры о мире, международные конфликты
- company: финансовые результаты конкретной компании, её дивидендах, руководстве или сделках
- skip: внутренняя политика, законодательство, спорт, технологии, культура, погода, криминал, социальные темы

Новость: {text}

Категория (одно слово):"""

VALID_LABELS = {"macro", "geo", "company", "skip"}

RU_MAP = {
    "макро": "macro", "макроэконом": "macro",
    "гео": "geo", "геополит": "geo",
    "компани": "company", "корпоратив": "company",
    "пропуст": "skip", "другое": "skip", "иное": "skip",
}

# ─── Ключевые слова (предфильтр) ─────────────────────────────────────────────

MACRO_KEYWORDS = [
    "ставк", "инфляц", "цб рф", "центробанк", "центральный банк",
    "ввп", "минфин", "федеральн бюджет", "государственн бюджет",
    "дефицит бюджет", "профицит бюджет", "расход бюджет", "доход бюджет",
    "urals", "брент", "brent", "нефть urals",
    "курс доллар", "курс евро", "курс рубл",
    "фрс", "федрезерв", "fomc", "денежно-кредитн",
    "торговый баланс", "безработиц",
    "офз", "облигаци федеральн",
    "ипотечн ставк", "рынок ипотек",
    "силуанов", "набиуллин",
    "экспорт нефт",
]

GEO_KEYWORDS = [
    "сво", "украин", "зеленск", "нато", "санкц", "переговор",
    "прекращени огня", "всу", "обстрел", "наступлени",
    "путин", "трамп", "байден", "тайвань",
    "израил", "газа", "ближний восток", "геополитик", "эмбарго",
]

RELEVANT_KEYWORDS = MACRO_KEYWORDS + GEO_KEYWORDS


def keyword_prefilter(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in RELEVANT_KEYWORDS)


# ─── Загрузка данных ─────────────────────────────────────────────────────────

def get_all_channels(db_path: str) -> list:
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'messages_%'", conn
        )["name"].tolist()
    return [t.removeprefix("messages_") for t in tables]


def load_messages(db_path: str, channel: str, limit=None) -> pd.DataFrame:
    table = f"messages_{channel}"
    query = (
        f"SELECT message_id, date, message, '{channel}' as channel "
        f"FROM {table} WHERE message IS NOT NULL "
        f"AND (tickers IS NULL OR tickers = '[]')"
    )
    if limit:
        query += f" ORDER BY date DESC LIMIT {limit}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df


# ─── Классификация одного текста ─────────────────────────────────────────────

def classify_one(text: str, llm, debug_count_ref: list) -> str:
    messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text[:1500])},
    ]
    result = llm.create_chat_completion(messages=messages, max_tokens=15, temperature=0.0)
    raw = result["choices"][0]["message"]["content"].strip().lower()

    if debug_count_ref[0] < 3:
        print(f"[DEBUG] raw='{raw}'")
        debug_count_ref[0] += 1

    word = re.split(r"[\s\n,.\-:]", raw)[0]
    if word in VALID_LABELS:
        return word
    found = next((l for l in VALID_LABELS if l in raw), None)
    if found:
        return found
    return next((en for ru, en in RU_MAP.items() if ru in raw), "skip")


# ─── Сохранение ──────────────────────────────────────────────────────────────

def _append_csv(rows: list, path: str, write_header: bool):
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=write_header)
    print(f"  [checkpoint] +{len(rows)} → {path}")


def _save_results(output: str, output_relevant: str):
    if not os.path.exists(output):
        print("Нет данных для сохранения.")
        return
    df_all = pd.read_csv(output)
    print(f"\n=== Распределение по категориям ===")
    print(df_all["category"].value_counts())
    df_rel = df_all[df_all["category"].isin(["macro", "geo"])]
    df_rel.to_csv(output_relevant, index=False)
    print(f"Только macro+geo: {len(df_rel)} → {output_relevant}")


# ─── Воркер для одного GPU ───────────────────────────────────────────────────

def worker(gpu_id: int, df_rows: list, out_csv: str):
    """
    Запускается в отдельном процессе.
    create_chat_completion не thread-safe → используем последовательный цикл.
    2x ускорение достигается за счёт двух независимых процессов на разных GPU.
    """
    # Назначаем GPU до инициализации CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Загрузка модели...")
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        n_ctx=1024,
        n_batch=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"[GPU {gpu_id}] Модель загружена. Задач: {len(df_rows):,}")

    # Пропускаем уже обработанные
    done_ids: set = set()
    if os.path.exists(out_csv):
        done_df = pd.read_csv(out_csv, usecols=["message_id"])
        done_ids = set(done_df["message_id"].tolist())
        print(f"[GPU {gpu_id}] Чекпоинт: уже обработано {len(done_ids):,}, пропускаем.")

    todo = [r for r in df_rows if r["message_id"] not in done_ids]
    print(f"[GPU {gpu_id}] Осталось: {len(todo):,}")

    buffer = []
    first_write = not os.path.exists(out_csv)
    debug_count = [0]

    for row in tqdm(todo, desc=f"GPU {gpu_id}", position=gpu_id):
        label = classify_one(row["message"], llm, debug_count)
        buffer.append({
            "message_id": row["message_id"],
            "channel":    row["channel"],
            "date":       row["date"],
            "message":    row["message"],
            "category":   label,
        })

        if len(buffer) >= CHECKPOINT_EVERY:
            _append_csv(buffer, out_csv, write_header=first_write)
            first_write = False
            buffer = []

    if buffer:
        _append_csv(buffer, out_csv, write_header=first_write)

    print(f"[GPU {gpu_id}] Готово.")


# ─── Главный пайплайн ────────────────────────────────────────────────────────

def main():
    # 1. Данные
    channels = get_all_channels(DB_PATH)
    print(f"Найдено каналов: {len(channels)}: {channels}")

    parts = [load_messages(DB_PATH, ch, limit=LIMIT) for ch in channels]
    df = pd.concat(parts, ignore_index=True)
    print(f"Итого загружено {len(df):,} сообщений")

    # 2. Предфильтр
    mask = df["message"].apply(keyword_prefilter)
    df_filtered = df[mask].copy().reset_index(drop=True)
    print(f"После предфильтра: {len(df_filtered):,} ({mask.mean():.1%})")

    if df_filtered.empty:
        print("Нет сообщений после фильтра.")
        return

    # 3. Пропускаем уже попавшие в финальный файл
    done_ids: set = set()
    if os.path.exists(OUTPUT):
        done_df = pd.read_csv(OUTPUT, usecols=["message_id"])
        done_ids = set(done_df["message_id"].tolist())
        print(f"Финальный CSV: уже {len(done_ids):,} записей.")

    # Также учитываем checkpoint-файлы воркеров
    for g in (0, 1):
        p = os.path.join(WORK_DIR, f"classified_gpu{g}.csv")
        if os.path.exists(p):
            tmp = pd.read_csv(p, usecols=["message_id"])
            done_ids.update(tmp["message_id"].tolist())

    df_todo = df_filtered[~df_filtered["message_id"].isin(done_ids)].copy().reset_index(drop=True)
    print(f"Осталось к обработке: {len(df_todo):,}")

    if df_todo.empty:
        print("Всё уже обработано — финализируем.")
        _merge_and_save()
        return

    # 4. Проверяем количество GPU
    import subprocess as _sp
    try:
        gpu_count = int(_sp.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().count("\n")) + 1
    except Exception:
        gpu_count = 1
    print(f"Обнаружено GPU: {gpu_count}")

    rows = df_todo.to_dict("records")

    if gpu_count >= 2:
        # 2-GPU режим: делим данные пополам
        mid = len(rows) // 2
        halves = [rows[:mid], rows[mid:]]
        out_csvs = [
            os.path.join(WORK_DIR, "classified_gpu0.csv"),
            os.path.join(WORK_DIR, "classified_gpu1.csv"),
        ]

        print(f"Запускаю 2 воркера: GPU0={len(halves[0]):,}, GPU1={len(halves[1]):,}")

        procs = []
        for g in range(2):
            p = mp.Process(target=worker, args=(g, halves[g], out_csvs[g]), daemon=True)
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        print("Оба воркера завершили работу.")
        _merge_and_save()

    else:
        # 1-GPU режим (fallback)
        print("Только 1 GPU — однопоточный режим.")
        out_csv = os.path.join(WORK_DIR, "classified_gpu0.csv")
        worker(0, rows, out_csv)
        _merge_and_save()


def _merge_and_save():
    """Объединяет classified_gpu*.csv → classified_news_yandex.csv, финальная статистика."""
    gpu_files = sorted(_glob.glob(os.path.join(WORK_DIR, "classified_gpu*.csv")))

    if not gpu_files:
        print("GPU-файлы не найдены, нечего мержить.")
        return

    parts = []
    # Добавляем уже существующий финальный файл (если есть)
    if os.path.exists(OUTPUT):
        parts.append(pd.read_csv(OUTPUT))

    for f in gpu_files:
        parts.append(pd.read_csv(f))

    df_all = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["message_id"])
    df_all.to_csv(OUTPUT, index=False)
    print(f"\nМерж завершён: {len(df_all):,} записей → {OUTPUT}")

    print("\n=== Распределение по категориям ===")
    print(df_all["category"].value_counts())

    df_rel = df_all[df_all["category"].isin(["macro", "geo"])]
    df_rel.to_csv(OUTPUT_RELEVANT, index=False)
    print(f"Только macro+geo: {len(df_rel):,} → {OUTPUT_RELEVANT}")


if __name__ == "__main__":
    # fork на Linux (Kaggle): дочерний процесс наследует всё из родителя,
    # CUDA ещё не инициализирована → безопасно.
    # spawn не работает в Kaggle/Jupyter (__main__ не является обычным модулем).
    mp.set_start_method("fork", force=True)
    main()
