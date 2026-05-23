import re
import json
import sqlite3
import os
from pathlib import Path

# Пути к файлам
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TICKERS_JSON = PROJECT_ROOT / "tickers.json"

NEWS_DBS = [
    {
        "path": SCRIPT_DIR / "commersant_news.db",
        "source": "commersant",
        "tables": ["Finansy", "economica", "politika", "Potrebrynok"],
    },
    {
        "path": SCRIPT_DIR / "lenta_news.db",
        "source": "lenta",
        "tables": ["economics", "economica", "world", "politika"],
    },
    {
        "path": SCRIPT_DIR / "ria_news.db",
        "source": "ria",
        "tables": ["economica", "politika"],
    },
]

# ──────────────────────────────────────────────────────────────
# Логика построения словаря компаний (из find_tickers.py)
# ──────────────────────────────────────────────────────────────

def load_tickers_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_case_variants(word: str) -> set:
    variants = {word}
    if not re.search(r"[а-яё]", word):
        return variants
    if len(word) > 3:
        if word.endswith(("к", "г", "х", "ж", "ч", "ш", "щ")):
            variants.add(word + "а")
        elif word.endswith(("ь", "й")):
            variants.add(word[:-1] + "я")
        elif word.endswith("а"):
            variants.add(word[:-1] + "ы")
            variants.add(word[:-1] + "и")
        elif word.endswith("я"):
            variants.add(word[:-1] + "и")
        else:
            variants.add(word + "а")
    if len(word) > 3:
        if word.endswith(("ь", "й")):
            variants.add(word[:-1] + "ю")
        elif word.endswith(("а", "я")):
            variants.add(word[:-1] + "е")
        else:
            variants.add(word + "у")
    if len(word) > 3:
        if word.endswith(("ь", "й")):
            variants.add(word[:-1] + "ем")
        elif word.endswith(("а", "я")):
            variants.add(word[:-1] + "ой")
        else:
            variants.add(word + "ом")
    if len(word) > 3:
        if word.endswith(("а", "я")):
            variants.add(word[:-1] + "е")
        else:
            variants.add(word + "е")
    return variants


def build_companies_dict(company_rows):
    companies = {}
    common_words = {
        "банк", "группа", "компания", "холдинг", "корпорация",
        "система", "центр", "фонд", "альянс", "концерн",
        "объединение", "ассоциация", "союз", "партнерство",
    }
    for row in company_rows:
        ticker = row["tiker"].upper()
        name = row["name"]
        sector = row.get("sector")
        name_lower = name.lower()
        alternative_names = []
        name_without_brackets = re.sub(
            r"\(([^)]+)\)",
            lambda m: (alternative_names.append(m.group(1).strip()), "")[1],
            name_lower,
        )
        name_clean = re.sub(r"[«»\"']", " ", name_without_brackets)
        name_clean = re.sub(r"[,/]", " ", name_clean)
        name_clean = re.sub(r"\s+", " ", name_clean).strip()
        tokens = name_clean.split()
        patterns = set()
        if name_clean:
            patterns.add(name_clean)
            for alt_name in alternative_names:
                alt_clean = re.sub(r"[«»\"']", " ", alt_name)
                alt_clean = re.sub(r"\s+", " ", alt_clean).strip()
                if alt_clean:
                    patterns.add(alt_clean)
                    for alt_token in alt_clean.split():
                        if len(alt_token) >= 4 and alt_token not in common_words:
                            alt_token_clean = re.sub(r"^[^\w-]+|[^\w-]+$", "", alt_token)
                            if len(alt_token_clean) >= 4:
                                patterns.add(alt_token_clean)
                                if re.search(r"[а-яё]", alt_token_clean):
                                    patterns.update(add_case_variants(alt_token_clean))
        if "," in name_lower or "/" in name_lower:
            for part in re.split(r"[,/]", name_lower):
                part_clean = re.sub(r"[«»\"'()]", " ", part)
                part_clean = re.sub(r"\s+", " ", part_clean).strip()
                if part_clean and len(part_clean) >= 4:
                    patterns.add(part_clean)
            if re.search(r"[а-яё]", name_clean) and len(tokens) > 1:
                last_word = tokens[-1]
                if last_word not in common_words and len(last_word) >= 4:
                    for variant in add_case_variants(last_word):
                        patterns.add(" ".join(tokens[:-1]) + " " + variant)
        if len(tokens) >= 1:
            first_word = tokens[0]
            if first_word not in common_words and len(first_word) >= 4:
                patterns.add(first_word)
                if re.search(r"[а-яё]", first_word):
                    patterns.update(add_case_variants(first_word))
        patterns.add(ticker.lower())
        companies[ticker] = {"sector": sector, "patterns": list(patterns)}
    return companies


# ──────────────────────────────────────────────────────────────
# Предобработка текста
# ──────────────────────────────────────────────────────────────

def prepare_for_match(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://[^\s]+", " ", text)
    text = re.sub(r"t\.me/[^\s]+", " ", text)
    text = re.sub(r"www\.[^\s]+", " ", text)
    text = re.sub(r"\b[a-z]\s*[+\-]\s*\d+\b", " ", text)
    text = re.sub(r"[^a-zа-я0-9#\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────
# Защита от ложных срабатываний специфичных для СМИ
# ──────────────────────────────────────────────────────────────

# Позитивные признаки компании «Самолет» (девелопер) — для SMLT
DEVELOPER_WORDS = {
    "девелопер", "застройщик", "ипотека", "жилье", "жильё", "квартир",
    "недвижимость", "новостройка", "жилой комплекс", "самолет груп",
    "группа самолет", "акции самолет",
}

# Позитивные признаки компании «Магнит» (ритейл) — для MGNT
RETAIL_WORDS = {
    "ритейл", "ритейлер", "супермаркет", "гипермаркет", "магазин",
    "торговая сеть", "продуктовый", "продуктов", "розничная", "розничный",
    "торговля продуктами", "акции магнит",
}


def _context_around(text_norm: str, match_start: int, match_end: int,
                    before_chars: int = 80, after_chars: int = 80) -> tuple[str, str]:
    before = text_norm[max(0, match_start - before_chars): match_start]
    after = text_norm[match_end: match_end + after_chars]
    return before, after


def _has_any_word(text: str, word_set: set) -> bool:
    return any(w in text for w in word_set)


def _is_valid_smlt(text_norm: str) -> bool:
    """True только если в тексте есть признаки компании-девелопера «Самолет»."""
    return _has_any_word(text_norm, DEVELOPER_WORDS)


def _is_valid_mgnt(text_norm: str) -> bool:
    """True только если в тексте есть признаки ритейлера «Магнит»."""
    return _has_any_word(text_norm, RETAIL_WORDS)


# ──────────────────────────────────────────────────────────────
# Основная функция поиска тикеров
# ──────────────────────────────────────────────────────────────

def find_tickers(title: str, text: str, companies: dict) -> list:
    """
    Ищет тикеры в title + text новостной статьи.
    Возвращает отсортированный список тикеров.
    """
    combined = f"{title or ''} {text or ''}"
    if not combined.strip():
        return []

    text_norm = prepare_for_match(combined)
    text_lower = combined.lower()
    tokens = text_norm.split()
    found = set()
    tickers_lower = [t.lower() for t in companies.keys()]

    # 1) Явные тикеры и хештеги
    for tok in tokens:
        tok_clean = tok.lstrip("#")
        if tok_clean not in tickers_lower:
            continue
        idx = tickers_lower.index(tok_clean)
        ticker = list(companies.keys())[idx]

        # X5 — исключаем автомобили
        if ticker == "X5":
            car_brands = {
                "bmw", "mercedes", "audi", "lexus", "porsche", "range rover",
                "land rover", "volvo", "kia", "hyundai", "toyota", "nissan",
                "honda", "mazda", "subaru", "infiniti",
            }
            company_indicators = {
                "group", "retail", "company", "компания", "группа", "акции",
                "дивиденды", "отчет", "отчёт", "выручка", "ebitda", "прибыль",
            }
            is_valid = False
            for m in re.finditer(r"\bx5\b", text_lower):
                before = text_lower[max(0, m.start() - 20): m.start()]
                after = text_lower[m.end(): m.end() + 30]
                if any(ind in after for ind in company_indicators):
                    is_valid = True
                    break
                if not any(b in before[-15:] for b in car_brands):
                    is_valid = True
                    break
            if not is_valid:
                continue

        # Короткие тикеры (1–2 символа, кроме T)
        if len(ticker) <= 2 and ticker != "T":
            is_valid = False
            for m in re.finditer(rf"\b{re.escape(ticker.lower())}\b", text_lower):
                after = text_lower[m.end(): m.end() + 3]
                before = text_lower[max(0, m.start() - 3): m.start()]
                if re.search(r"[+\-]\s*\d+", after):
                    continue
                if after.startswith(".me") or before.endswith("t."):
                    continue
                if after.startswith("5") and not before:
                    continue
                is_valid = True
                break
            if not is_valid:
                continue

        found.add(ticker)

    # 2) Поиск по именам компаний
    for ticker, info in companies.items():
        if len(ticker) <= 2 and ticker != "T":
            continue

        for p in info["patterns"]:
            if p == ticker.lower():
                continue
            if re.search(rf"\b{re.escape(p)}\b", text_norm):
                # LENT — исключаем «лента новостей» и «lenta.ru» / «ленты.ру»
                if ticker == "LENT":
                    news_feed_words = {
                        "новостей", "новостная", "новостной", "информации",
                        "сообщений", "публикаций",
                    }
                    is_valid = False
                    for m in re.finditer(rf"\b{re.escape(p)}\b", text_norm):
                        before = text_norm[max(0, m.start() - 20): m.start()]
                        after = text_norm[m.end(): m.end() + 10]
                        # «ленты ру», «лента ру» — ссылка на сайт lenta.ru
                        if re.match(r"\s*ру\b", after):
                            continue
                        if any(w in after for w in news_feed_words):
                            continue
                        if "новостная" in before[-15:]:
                            continue
                        is_valid = True
                        break
                    if not is_valid:
                        continue

                found.add(ticker)
                break

    # 3) Специальные проверки для неоднозначных имён
    if "SMLT" in found and not _is_valid_smlt(text_norm):
        found.discard("SMLT")

    if "MGNT" in found and not _is_valid_mgnt(text_norm):
        found.discard("MGNT")

    return sorted(found)


# ──────────────────────────────────────────────────────────────
# Создание выходной БД
# ──────────────────────────────────────────────────────────────

OUTPUT_DB = SCRIPT_DIR / "news_with_tickers.db"

def create_output_db(out_conn: sqlite3.Connection):
    """Создаёт таблицу news_ticker в выходной БД."""
    out_conn.execute("""
        CREATE TABLE IF NOT EXISTS news_ticker (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            source               TEXT NOT NULL,
            section              TEXT NOT NULL,
            ticker               TEXT NOT NULL,
            news_id              TEXT NOT NULL,
            publication_datetime TEXT NOT NULL,
            url                  TEXT,
            title                TEXT,
            text                 TEXT,
            tags                 TEXT
        )
    """)
    out_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nt_ticker_date
        ON news_ticker (ticker, publication_datetime)
    """)
    out_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nt_date
        ON news_ticker (publication_datetime)
    """)
    out_conn.commit()


def process_table(
    src_conn: sqlite3.Connection,
    out_conn: sqlite3.Connection,
    source: str,
    section: str,
    companies: dict,
) -> int:
    """
    Читает таблицу источника, ищет тикеры, записывает в news_ticker.
    Одна строка на (новость × тикер).
    Возвращает кол-во записанных строк.
    """
    rows = src_conn.execute(
        f'SELECT news_id, publication_datetime, url, title, text, tags '
        f'FROM "{section}" '
        f'WHERE text IS NOT NULL OR title IS NOT NULL'
    ).fetchall()
    print(f"   Строк в источнике: {len(rows)}")

    batch = []
    for news_id, pub_dt, url, title, text, tags in rows:
        tickers = find_tickers(title or "", text or "", companies)
        for ticker in tickers:
            batch.append((
                source, section, ticker,
                news_id, pub_dt, url, title, text, tags,
            ))

    if batch:
        out_conn.executemany(
            """INSERT OR IGNORE INTO news_ticker
               (source, section, ticker, news_id, publication_datetime,
                url, title, text, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )
        out_conn.commit()

    print(f"   Записано строк (новость×тикер): {len(batch)}")
    return len(batch)


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ПОИСК ТИКЕРОВ В НОВОСТНЫХ ЛЕНТАХ")
    print(f"   Результат → {OUTPUT_DB}")
    print("=" * 60)

    if not TICKERS_JSON.exists():
        print(f"Не найден {TICKERS_JSON}")
        return

    company_rows = load_tickers_from_json(TICKERS_JSON)
    companies = build_companies_dict(company_rows)
    print(f"Загружено компаний: {len(companies)}")

    out_conn = sqlite3.connect(OUTPUT_DB)
    create_output_db(out_conn)

    total_rows = 0

    for db_info in NEWS_DBS:
        db_path = db_info["path"]
        source = db_info["source"]
        tables = db_info["tables"]

        print(f"\n{'=' * 60}")
        print(f"Источник: {source}  ({db_path.name})")
        print(f"{'=' * 60}")

        if not db_path.exists():
            print(f"   Файл не найден, пропускаю")
            continue

        src_conn = sqlite3.connect(db_path)
        existing = {
            row[0]
            for row in src_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        for table in tables:
            if table not in existing:
                print(f"   Таблица {table} не найдена, пропускаю")
                continue
            print(f"\n   Раздел: {table}")
            total_rows += process_table(src_conn, out_conn, source, table, companies)

        src_conn.close()

    # Финальная сортировка: пересоздаём таблицу в порядке даты
    print(f"\n⏳ Сортировка по дате...")
    out_conn.execute("""
        CREATE TABLE IF NOT EXISTS news_ticker_sorted AS
        SELECT * FROM news_ticker
        ORDER BY publication_datetime ASC
    """)
    out_conn.execute("DROP TABLE news_ticker")
    out_conn.execute("ALTER TABLE news_ticker_sorted RENAME TO news_ticker")
    out_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nt_ticker_date
        ON news_ticker (ticker, publication_datetime)
    """)
    out_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nt_date
        ON news_ticker (publication_datetime)
    """)
    out_conn.commit()

    total_in_db = out_conn.execute("SELECT COUNT(*) FROM news_ticker").fetchone()[0]
    out_conn.close()

    print(f"\n{'=' * 60}")
    print(f"ГОТОВО")
    print(f"   Записей (новость×тикер): {total_rows:,}")
    print(f"   Итого в БД:              {total_in_db:,}")
    print(f"   Файл: {OUTPUT_DB}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
