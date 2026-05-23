"""
pipeline_classify_signal.py

Объединённый пайплайн:
  1. Классификация всех сообщений: macro / geo / company / skip
  2. Для macro/geo — сразу анализ scope (market/sector) и affected_sectors/tickers

Одна загрузка GGUF-модели, один проход по данным.

Выходные файлы:
  classified_news_yandex.csv   — все классифицированные сообщения
  macro_geo_signals.csv        — только macro/geo со scope и секторами
"""

import os
import subprocess
import sys
import json
import sqlite3
import re
import glob as _glob

import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# llama-cpp-python с CUDA
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

# ─── Пути ────────────────────────────────────────────────────────────────────

_DB_CANDIDATES = [
    "/kaggle/input/datasets/markbabii/telegram-messages-new/telegram_messages_new.db",
    "/kaggle/input/telegram-messages-new/telegram_messages_new.db",
    "/kaggle/input/telegram-messages-2/telegram_messages.db",
    "/kaggle/input/telegram-messages/telegram_messages.db",
    "/Users/markbabii/PycharmProjects/analiz_santiment/telegram_messages_new.db",
    "telegram_messages_new.db",
]
DB_PATH = next((p for p in _DB_CANDIDATES if os.path.exists(p)), None)
if DB_PATH is None:
    _found = _glob.glob("/kaggle/input/**/*.db", recursive=True)
    if _found:
        DB_PATH = _found[0]
    else:
        raise FileNotFoundError(f"telegram_messages.db не найдена.\nПробовали: {_DB_CANDIDATES}")

_TICKERS_CANDIDATES = [
    "/kaggle/input/tickers/tickers.json",
    "/kaggle/input/telegram-messages-db/tickers.json",
    "/kaggle/working/tickers.json",
    "/Users/markbabii/PycharmProjects/analiz_santiment/tickers.json",
    "tickers.json",
    "../tickers.json",
]
TICKERS_PATH = next((p for p in _TICKERS_CANDIDATES if os.path.exists(p)), None)
if TICKERS_PATH is None:
    _found = _glob.glob("/kaggle/input/**/tickers.json", recursive=True)
    TICKERS_PATH = _found[0] if _found else None

WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

OUTPUT_CLASSIFIED = os.path.join(WORK_DIR, "classified_news_yandex.csv")
OUTPUT_SIGNALS    = os.path.join(WORK_DIR, "macro_geo_signals.csv")

print(f"БД: {DB_PATH}")
if TICKERS_PATH:
    print(f"Тикеры найдены: {TICKERS_PATH}")
else:
    print("ВНИМАНИЕ: tickers.json не найден — signal-анализ будет пропущен.")

# ─── Настройки ───────────────────────────────────────────────────────────────

LIMIT            = None
CHECKPOINT_EVERY = 500

MODEL_REPO     = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF"
MODEL_FILENAME = "YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"

# ─── Выбор каналов ───────────────────────────────────────────────────────────

CHANNELS_SMALL = [
    "headlines_quants",        # 1,210
    "headlines_geo",           # 2,452
    "Dividend_News100",        # 4,191
    "headlines_MACRO",         # 4,372
    "if_stocks",               # 6,792
    "econs",                   # 7,658
    "investfuture",            # 9,053
    "sosisochniyparserru",     # 13,258
    "FatCat18",                # 14,744
    "AK47pfl",                 # 14,970
    "frank_media",             # 17,702
    "economika",               # 21,986
    "bankser",                 # 23,369
    "investingcorp",           # 25,266
    "BIoomberg",               # 37,811
]

CHANNELS_LARGE = [
    "vedomosti",               # 56,306
    "Stock_News100",           # 62,105
    "headlines_for_traders",   # 62,341
    "banksta",                 # 64,489
    "cbrstocks",               # 68,787
    "if_market_news",          # 70,042
    "forbesrussia",            # 71,548
    "kommersant",              # 76,340
    "newssmartlab",            # 102,558
    "rbc_news",                # 116,917
    "rt_russian",              # 209,529
    "markettwits",             # 276,665
    "rian_ru",                 # 280,501
    "tass_agency",             # 321,883
]

# [] = все каналы из telegram_messages_new.db
CHANNELS_TO_PROCESS = [
    "AK47pfl",
    "BIoomberg",
    "Dividend_News100",
    "FatCat18",
    "bankser",
    "economika",
    "econs",
    "frank_media",
    "headlines_MACRO",
    "headlines_for_traders",
    "headlines_geo",
    "headlines_quants",
    "if_market_news",
    "if_stocks",
    "investfuture",
    "investingcorp",
    "sosisochniyparserru",
    "vedomosti",
]

# ─── Промпты ─────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """Определи категорию новости для анализа российского фондового рынка (MOEX). Ответь ТОЛЬКО одним словом из списка: macro, geo, company, skip. Никаких объяснений.

Категории:
- macro: ключевая ставка ЦБ, инфляция, ВВП, бюджет, курс рубля, цены на нефть как экономический индикатор
- geo: война на Украине, международные санкции, военные действия, переговоры о мире, международные конфликты
- company: финансовые результаты конкретной компании, её дивидендах, руководстве или сделках
- skip: внутренняя политика, законодательство, спорт, технологии, культура, погода, криминал, социальные темы

Новость: {text}

Категория (одно слово):"""

SIGNAL_PROMPT = """Ты анализируешь новости для управления портфелем российских акций (MOEX).

Доступные секторы:
{sectors_list}

Правила:
- scope="market" ТОЛЬКО если новость одновременно влияет на ВСЕ или почти все секторы (например: начало войны, дефолт, резкий обвал рубля, глобальный кризис)
- scope="sector" во всех остальных случаях — даже если затронуто 3-5 секторов
- sectors — перечисли ТОЛЬКО реально затронутые секторы через запятую (точно из списка выше)

Примеры:
- "Нефть Urals упала до $37" → scope: sector / sectors: Нефть и газ
- "Санкции на теневой флот" → scope: sector / sectors: Морские перевозки, Нефть и газ
- "ЦБ поднял ставку до 21%" → scope: sector / sectors: Банки, Девелопмент
- "Россия начала СВО" → scope: market / sectors: все

Отвечай строго в формате (2 строки, без пояснений):
scope: market/sector
sectors: Банки, Нефть и газ

Новость: {text}"""

# ─── Метки классификации ─────────────────────────────────────────────────────

VALID_LABELS = {"macro", "geo", "company", "skip"}

RU_MAP = {
    "макро": "macro", "макроэконом": "macro",
    "гео": "geo", "геополит": "geo",
    "компани": "company", "корпоратив": "company",
    "пропуст": "skip", "другое": "skip", "иное": "skip",
}

# ─── Ключевые слова ───────────────────────────────────────────────────────────

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
        return pd.read_sql_query(query, conn)


def load_tickers(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sector_to_tickers: dict = {}
    for item in data:
        sector = item["sector"]
        ticker = item["tiker"]
        sector_to_tickers.setdefault(sector, []).append(ticker)
    return sector_to_tickers


# ─── Инференс ────────────────────────────────────────────────────────────────

def classify_one(text: str, llm, debug_ref: list) -> str:
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:1500])}],
        max_tokens=15, temperature=0.0
    )
    raw = result["choices"][0]["message"]["content"].strip().lower()

    if debug_ref[0] < 3:
        print(f"  [classify] raw='{raw}'")
        debug_ref[0] += 1

    word = re.split(r"[\s\n,.\-:]", raw)[0]
    if word in VALID_LABELS:
        return word
    found = next((l for l in VALID_LABELS if l in raw), None)
    if found:
        return found
    return next((en for ru, en in RU_MAP.items() if ru in raw), "skip")


def parse_signal(raw: str, valid_sectors: set, all_sectors: list) -> dict:
    result = {"scope": "sector", "sectors": []}
    for line in raw.lower().splitlines():
        line = line.strip()
        if line.startswith("scope:"):
            val = line.split(":", 1)[1].strip()
            result["scope"] = "market" if ("market" in val or "весь" in val or "все" in val) else "sector"
        elif line.startswith("sectors:"):
            val = line.split(":", 1)[1].strip()
            if "все" in val or "all" in val:
                result["sectors"] = all_sectors
            else:
                result["sectors"] = [s for s in valid_sectors if s.lower() in val]
    if result["scope"] == "market" and not result["sectors"]:
        result["sectors"] = all_sectors
    return result


def signal_one(text: str, llm, sectors_list: str,
               valid_sectors: set, all_sectors: list,
               sector_to_tickers: dict) -> dict:
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": SIGNAL_PROMPT.format(
            sectors_list=sectors_list, text=text[:1200]
        )}],
        max_tokens=60, temperature=0.0
    )
    raw = result["choices"][0]["message"]["content"].strip()
    parsed = parse_signal(raw, valid_sectors, all_sectors)

    if parsed["scope"] == "market":
        return {"scope": "market", "affected_sectors": "", "affected_tickers": ""}
    else:
        affected_sectors = ", ".join(parsed["sectors"])
        affected_tickers = ", ".join(sorted(set(
            t for s in parsed["sectors"] for t in sector_to_tickers.get(s, [])
        )))
        return {"scope": "sector", "affected_sectors": affected_sectors, "affected_tickers": affected_tickers}


# ─── Сохранение ──────────────────────────────────────────────────────────────

def _append_csv(rows: list, path: str, write_header: bool):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=write_header)
        print(f"  [checkpoint] +{len(rows)} → {path}")


# ─── Воркер: Фаза 1 — Классификация ─────────────────────────────────────────

def worker_classify(gpu_id: int, df_rows: list, out_csv: str):
    """Быстрая классификация: 1 LLM вызов на сообщение, ~8-9 it/s."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] [Фаза 1] Загрузка модели...")
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        n_ctx=2048,
        n_batch=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"[GPU {gpu_id}] Модель загружена. Задач: {len(df_rows):,}")

    done_ids: set = set()
    if os.path.exists(out_csv):
        tmp = pd.read_csv(out_csv, usecols=["message_id", "channel"])
        done_ids = set(zip(tmp["message_id"], tmp["channel"]))
        print(f"[GPU {gpu_id}] Чекпоинт: {len(done_ids):,} уже обработано.")

    todo = [r for r in df_rows if (r["message_id"], r["channel"]) not in done_ids]
    print(f"[GPU {gpu_id}] Осталось: {len(todo):,}")

    buffer = []
    first_write = not os.path.exists(out_csv)
    debug_ref = [0]

    for row in tqdm(todo, desc=f"Classify GPU{gpu_id}", position=gpu_id):
        category = classify_one(row["message"], llm, debug_ref)
        buffer.append({
            "message_id": row["message_id"],
            "channel":    row["channel"],
            "date":       row["date"],
            "message":    row["message"],
            "category":   category,
        })
        if len(buffer) >= CHECKPOINT_EVERY:
            _append_csv(buffer, out_csv, write_header=first_write)
            first_write = False
            buffer = []

    if buffer:
        _append_csv(buffer, out_csv, write_header=first_write)
    print(f"[GPU {gpu_id}] [Фаза 1] Готово.")


# ─── Воркер: Фаза 2 — Signal-анализ ─────────────────────────────────────────

def worker_signal(gpu_id: int, df_rows: list, out_csv: str, tickers_path: str):
    """Signal-анализ только для macro/geo. Промпт длиннее → n_ctx=1536."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sector_to_tickers = load_tickers(tickers_path)
    valid_sectors = set(sector_to_tickers.keys())
    all_sectors = sorted(valid_sectors)
    sectors_list = "\n".join(f"- {s}" for s in all_sectors)
    print(f"[GPU {gpu_id}] [Фаза 2] Секторов: {len(valid_sectors)}")

    print(f"[GPU {gpu_id}] [Фаза 2] Загрузка модели...")
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        n_ctx=1536,   # signal промпт длиннее: ~800 токенов + 60 вывод
        n_batch=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"[GPU {gpu_id}] Модель загружена. Задач: {len(df_rows):,}")

    done_ids: set = set()
    if os.path.exists(out_csv):
        tmp = pd.read_csv(out_csv, usecols=["message_id", "channel"])
        done_ids = set(zip(tmp["message_id"], tmp["channel"]))
        print(f"[GPU {gpu_id}] Чекпоинт: {len(done_ids):,} уже обработано.")

    todo = [r for r in df_rows if (r["message_id"], r["channel"]) not in done_ids]
    print(f"[GPU {gpu_id}] Осталось: {len(todo):,}")

    buffer = []
    first_write = not os.path.exists(out_csv)

    for row in tqdm(todo, desc=f"Signal GPU{gpu_id}", position=gpu_id):
        sig = signal_one(row["message"], llm, sectors_list, valid_sectors, all_sectors, sector_to_tickers)
        buffer.append({
            "message_id":       row["message_id"],
            "channel":          row["channel"],
            "date":             row["date"],
            "category":         row["category"],
            "message":          row["message"],
            "scope":            sig["scope"],
            "affected_sectors": sig["affected_sectors"],
            "affected_tickers": sig["affected_tickers"],
        })
        if len(buffer) >= CHECKPOINT_EVERY:
            _append_csv(buffer, out_csv, write_header=first_write)
            first_write = False
            buffer = []

    if buffer:
        _append_csv(buffer, out_csv, write_header=first_write)
    print(f"[GPU {gpu_id}] [Фаза 2] Готово.")


# ─── Хелпер: запуск 2-GPU или 1-GPU ─────────────────────────────────────────

def _run_parallel(target, rows, out_files, extra_args=()):
    import subprocess as _sp
    try:
        gpu_count = _sp.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().count("\n") + 1
    except Exception:
        gpu_count = 1

    if gpu_count >= 2:
        mid = len(rows) // 2
        halves = [rows[:mid], rows[mid:]]
        print(f"2-GPU: GPU0={len(halves[0]):,}, GPU1={len(halves[1]):,}")
        procs = [
            mp.Process(target=target, args=(g, halves[g], out_files[g]) + extra_args, daemon=True)
            for g in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    else:
        print("1-GPU режим.")
        target(0, rows, out_files[0], *extra_args)


# ─── Главный пайплайн ────────────────────────────────────────────────────────

def main():
    # ── Загрузка данных ──
    all_channels = get_all_channels(DB_PATH)
    if CHANNELS_TO_PROCESS:
        channels = [ch for ch in CHANNELS_TO_PROCESS if ch in all_channels]
        missing  = [ch for ch in CHANNELS_TO_PROCESS if ch not in all_channels]
        if missing:
            print(f"ВНИМАНИЕ: каналы не найдены в БД: {missing}")
    else:
        channels = all_channels
    print(f"Обрабатываем {len(channels)} каналов: {channels}")

    df = pd.concat(
        [load_messages(DB_PATH, ch, limit=LIMIT) for ch in channels],
        ignore_index=True
    )
    print(f"Итого сообщений: {len(df):,}")

    mask = df["message"].apply(keyword_prefilter)
    df_filtered = df[mask].copy().reset_index(drop=True)
    print(f"После предфильтра: {len(df_filtered):,} ({mask.mean():.1%})")

    if df_filtered.empty:
        print("Нет данных.")
        return

    # ══════════════════════════════════════════════════════════
    # ФАЗА 1: Классификация
    # ══════════════════════════════════════════════════════════
    done_ids: set = set()
    for path in [OUTPUT_CLASSIFIED] + [
        os.path.join(WORK_DIR, f"classified_gpu{g}.csv") for g in (0, 1)
    ]:
        if os.path.exists(path):
            tmp = pd.read_csv(path, usecols=["message_id", "channel"])
            done_ids.update(zip(tmp["message_id"], tmp["channel"]))

    df_todo = df_filtered[
        ~pd.Series(list(zip(df_filtered["message_id"], df_filtered["channel"])), index=df_filtered.index).isin(done_ids)
    ].copy().reset_index(drop=True)
    print(f"\n═══ ФАЗА 1: Классификация ═══")
    print(f"Уже классифицировано: {len(done_ids):,} | Осталось: {len(df_todo):,}")

    if not df_todo.empty:
        out_cls = [os.path.join(WORK_DIR, f"classified_gpu{g}.csv") for g in range(2)]
        _run_parallel(worker_classify, df_todo.to_dict("records"), out_cls)

    # Мерж classify файлов → OUTPUT_CLASSIFIED
    cls_files = sorted(_glob.glob(os.path.join(WORK_DIR, "classified_gpu*.csv")))
    parts = ([pd.read_csv(OUTPUT_CLASSIFIED)] if os.path.exists(OUTPUT_CLASSIFIED) else [])
    parts += [pd.read_csv(f) for f in cls_files]
    df_cls = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["message_id", "channel"])
    df_cls.to_csv(OUTPUT_CLASSIFIED, index=False)
    print(f"\nClassified: {len(df_cls):,} → {OUTPUT_CLASSIFIED}")
    print(df_cls["category"].value_counts())

    # ══════════════════════════════════════════════════════════
    # ФАЗА 2: Signal-анализ (только macro/geo)
    # ══════════════════════════════════════════════════════════
    if not TICKERS_PATH:
        print("\ntickers.json не найден — signal-анализ пропущен.")
        return

    print(f"\n═══ ФАЗА 2: Signal-анализ ═══")
    df_macro_geo = df_cls[df_cls["category"].isin(["macro", "geo"])].copy()
    print(f"macro/geo новостей: {len(df_macro_geo):,}")

    done_sig: set = set()
    for path in [OUTPUT_SIGNALS] + [
        os.path.join(WORK_DIR, f"signals_gpu{g}.csv") for g in (0, 1)
    ]:
        if os.path.exists(path):
            tmp = pd.read_csv(path, usecols=["message_id", "channel"])
            done_sig.update(zip(tmp["message_id"], tmp["channel"]))

    df_sig_todo = df_macro_geo[
        ~pd.Series(list(zip(df_macro_geo["message_id"], df_macro_geo["channel"])), index=df_macro_geo.index).isin(done_sig)
    ].copy().reset_index(drop=True)
    print(f"Уже обработано: {len(done_sig):,} | Осталось: {len(df_sig_todo):,}")

    if not df_sig_todo.empty:
        out_sig = [os.path.join(WORK_DIR, f"signals_gpu{g}.csv") for g in range(2)]
        _run_parallel(worker_signal, df_sig_todo.to_dict("records"), out_sig, extra_args=(TICKERS_PATH,))

    # Мерж signal файлов → OUTPUT_SIGNALS
    sig_files = sorted(_glob.glob(os.path.join(WORK_DIR, "signals_gpu*.csv")))
    parts = ([pd.read_csv(OUTPUT_SIGNALS)] if os.path.exists(OUTPUT_SIGNALS) else [])
    parts += [pd.read_csv(f) for f in sig_files]
    df_sig = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["message_id", "channel"])
    df_sig.to_csv(OUTPUT_SIGNALS, index=False)
    print(f"\nSignals: {len(df_sig):,} → {OUTPUT_SIGNALS}")
    print(df_sig["scope"].value_counts())


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
