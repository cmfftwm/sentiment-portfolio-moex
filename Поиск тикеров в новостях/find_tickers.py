import re
import json
import sqlite3
import os
from pathlib import Path

# Пути к файлам
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TICKERS_JSON = PROJECT_ROOT / "tickers.json"
DB_PATH = PROJECT_ROOT / "telegram_messages_new.db"


def load_tickers_from_json(json_path):
    """Загружает список компаний из JSON файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_case_variants(word: str) -> set:
    """Добавляет основные падежные формы для русского слова"""
    variants = {word}
    
    # Только для русских слов (содержат кириллицу)
    if not re.search(r'[а-яё]', word):
        return variants
    
    # Основные окончания для падежей (упрощенная версия)
    # Родительный падеж: -а, -я, -ы, -и
    if len(word) > 3:
        if word.endswith(('к', 'г', 'х', 'ж', 'ч', 'ш', 'щ')):
            variants.add(word + 'а')
        elif word.endswith(('ь', 'й')):
            variants.add(word[:-1] + 'я')
        elif word.endswith('а'):
            # Для слов женского рода на -а: родительный падеж обычно -ы
            variants.add(word[:-1] + 'ы')
            # Также добавляем вариант с -и (для некоторых слов)
            variants.add(word[:-1] + 'и')
        elif word.endswith('я'):
            variants.add(word[:-1] + 'и')
        else:
            variants.add(word + 'а')
    
    # Дательный падеж: -у, -ю
    if len(word) > 3:
        if word.endswith(('ь', 'й')):
            variants.add(word[:-1] + 'ю')
        elif word.endswith(('а', 'я')):
            variants.add(word[:-1] + 'е')
        else:
            variants.add(word + 'у')
    
    # Творительный падеж: -ом, -ем, -ой
    if len(word) > 3:
        if word.endswith(('ь', 'й')):
            variants.add(word[:-1] + 'ем')
        elif word.endswith(('а', 'я')):
            variants.add(word[:-1] + 'ой')
        else:
            variants.add(word + 'ом')
    
    # Предложный падеж: -е, -и
    if len(word) > 3:
        if word.endswith(('а', 'я')):
            variants.add(word[:-1] + 'е')
        else:
            variants.add(word + 'е')
    
    return variants


def build_companies_dict(company_rows):
    """Строит словарь компаний с паттернами для поиска"""
    companies = {}
    
    # Список слишком общих слов, которые не должны использоваться как паттерны
    # (чтобы избежать ложных срабатываний)
    common_words = {
        "банк", "группа", "компания", "холдинг", "корпорация",
        "система", "центр", "фонд", "альянс", "концерн",
        "объединение", "ассоциация", "союз", "партнерство"
    }

    for row in company_rows:
        ticker = row["tiker"].upper()
        name = row["name"]
        sector = row.get("sector")

        # приводим имя к нижнему регистру и убираем кавычки
        name_lower = name.lower()
        
        # Извлекаем альтернативные названия из скобок
        alternative_names = []
        name_without_brackets = re.sub(r"\(([^)]+)\)", lambda m: (alternative_names.append(m.group(1).strip()), "")[1], name_lower)
        
        # Обрабатываем основное название
        name_clean = re.sub(r"[«»\"']", " ", name_without_brackets)
        # Заменяем запятые и слэши на пробелы (для разделения альтернативных названий)
        name_clean = re.sub(r"[,/]", " ", name_clean)
        name_clean = re.sub(r"\s+", " ", name_clean).strip()

        tokens = name_clean.split()
        patterns = set()

        if name_clean:
            patterns.add(name_clean)     # полное название
            # Добавляем альтернативные названия из скобок как отдельные паттерны
            for alt_name in alternative_names:
                alt_clean = re.sub(r"[«»\"']", " ", alt_name)
                alt_clean = re.sub(r"\s+", " ", alt_clean).strip()
                if alt_clean:
                    patterns.add(alt_clean)
                    # Разбиваем альтернативное название на токены и добавляем их
                    alt_tokens = alt_clean.split()
                    for alt_token in alt_tokens:
                        if len(alt_token) >= 4 and alt_token not in common_words:
                            alt_token_clean = re.sub(r'^[^\w-]+|[^\w-]+$', '', alt_token)
                            if len(alt_token_clean) >= 4:
                                patterns.add(alt_token_clean)
                                if re.search(r'[а-яё]', alt_token_clean):
                                    case_variants = add_case_variants(alt_token_clean)
                                    patterns.update(case_variants)
        
        # Если название содержит запятые или слэши, добавляем отдельные части
        # Например, "Т-банк, Т-Технологии" -> ["т-банк", "т-технологии"]
        if ',' in name_lower or '/' in name_lower:
            parts = re.split(r'[,/]', name_lower)
            for part in parts:
                part_clean = re.sub(r"[«»\"'()]", " ", part)
                part_clean = re.sub(r"\s+", " ", part_clean).strip()
                if part_clean and len(part_clean) >= 4:
                    patterns.add(part_clean)
            # Добавляем варианты полного названия с падежами (только для русских названий)
            if re.search(r'[а-яё]', name_clean):
                # Для многословных названий добавляем варианты последнего слова
                if len(tokens) > 1:
                    last_word = tokens[-1]
                    if last_word not in common_words and len(last_word) >= 4:
                        case_variants = add_case_variants(last_word)
                        for variant in case_variants:
                            # Создаем варианты с последним словом в разных падежах
                            base = ' '.join(tokens[:-1])
                            patterns.add(f"{base} {variant}")
        
        # Первое слово добавляем только если оно не слишком общее
        # и достаточно уникальное (минимум 4 символа)
        if len(tokens) >= 1:
            first_word = tokens[0]
            # Используем первое слово только если:
            # 1. Оно не в списке общих слов И
            # 2. Оно достаточно длинное (>= 4 символа)
            if first_word not in common_words and len(first_word) >= 4:
                patterns.add(first_word)
                # Добавляем падежные формы первого слова
                if re.search(r'[а-яё]', first_word):
                    case_variants = add_case_variants(first_word)
                    patterns.update(case_variants)
        
        patterns.add(ticker.lower())     # сам тикер, типа "afks"

        companies[ticker] = {
            "sector": sector,
            "patterns": list(patterns),
        }

    return companies


def prepare_for_match(text: str) -> str:
    """Нормализует текст для поиска"""
    text = str(text).lower()
    
    # Удаляем URL перед нормализацией, чтобы избежать ложных срабатываний
    # (например, "t.me" не должен находить тикер "T")
    text = re.sub(r"https?://[^\s]+", " ", text)  # http:// и https://
    text = re.sub(r"t\.me/[^\s]+", " ", text)     # t.me/...
    text = re.sub(r"www\.[^\s]+", " ", text)      # www.example.com
    
    # Удаляем выражения типа "T+1", "T-1", "t+1" и т.д. (торговые дни, не тикеры)
    # Паттерн: одна буква, затем знак + или -, затем цифра
    text = re.sub(r"\b[a-z]\s*[+\-]\s*\d+\b", " ", text)
    
    # оставляем буквы, цифры, #, дефис и пробел (дефис нужен для названий типа "Т-банк")
    text = re.sub(r"[^a-zа-я0-9#\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_tickers(text: str, companies: dict) -> list:
    """Находит тикеры в тексте"""
    if not text:
        return []
    
    text_norm = prepare_for_match(text)
    found = set()
    tokens = text_norm.split()

    # 1) явные тикеры и хештеги (#afks, afks)
    tickers_lower = [t.lower() for t in companies.keys()]
    text_lower = text.lower()
    
    for i, tok in enumerate(tokens):
        tok_clean = tok.lstrip("#")
        if tok_clean in tickers_lower:
            # восстанавливаем нормальный верхний регистр тикера
            idx = tickers_lower.index(tok_clean)
            ticker = list(companies.keys())[idx]
            
            # Специальная проверка для тикера X5 - исключаем упоминания автомобилей
            if ticker == "X5":
                ticker_lower = ticker.lower()
                # Ищем все вхождения "x5" в исходном тексте
                pattern = rf"\b{re.escape(ticker_lower)}\b"
                matches = list(re.finditer(pattern, text_lower))
                
                # Проверяем каждое вхождение
                is_valid = False
                for match in matches:
                    start, end = match.span()
                    # Проверяем контекст перед "x5" (до 20 символов назад)
                    before = text_lower[max(0, start-20):start]
                    # Проверяем контекст после "x5" (до 20 символов вперед)
                    after = text_lower[end:min(len(text_lower), end+20)]
                    
                    # Список марок автомобилей и связанных слов, которые указывают на автомобиль
                    car_brands = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'range rover', 
                                 'land rover', 'volvo', 'kia', 'hyundai', 'toyota', 'nissan',
                                 'honda', 'mazda', 'subaru', 'infiniti', 'acura', 'cadillac',
                                 'lincoln', 'jeep', 'ford', 'chevrolet', 'dodge', 'chrysler']
                    
                    # Слова, указывающие на компанию (если после x5 идет одно из этих слов, это компания)
                    company_indicators = ['group', 'retail', 'company', 'компания', 'группа', 
                                         'акции', 'дивиденды', 'отчет', 'отчёт', 'выручка',
                                         'ebitda', 'прибыль', 'руководство', 'совет директоров']
                    
                    # Проверяем, не является ли это компанией (приоритет над автомобилем)
                    is_company_mention = False
                    for indicator in company_indicators:
                        # Проверяем, что после x5 идет слово-индикатор компании
                        if re.search(rf'\b{re.escape(indicator)}\b', after[:30]):
                            is_company_mention = True
                            break
                    
                    # Если это точно компания, пропускаем проверку на автомобиль
                    if is_company_mention:
                        is_valid = True
                        break
                    
                    # Пропускаем, если перед "x5" стоит марка автомобиля
                    is_car_mention = False
                    for brand in car_brands:
                        if brand in before[-15:]:  # Проверяем последние 15 символов перед x5
                            is_car_mention = True
                            break
                    
                    # Также пропускаем, если после "x5" идет описание автомобиля
                    # (например, "x5 с объёмом", "x5 двигатель", "x5 л.с.", "x5 цена")
                    car_indicators = ['с объёмом', 'объёмом двигателя', 'двигатель', 'л.с.', 
                                     'лс', 'л/с', 'мощность', 'цена', 'стоимость', 'млн руб',
                                     'млн рублей', 'рублей', 'руб', 'модель', 'версия', 'комплектация']
                    
                    if not is_car_mention:
                        for indicator in car_indicators:
                            if indicator in after[:30]:  # Проверяем первые 30 символов после x5
                                is_car_mention = True
                                break
                    
                    if is_car_mention:
                        continue  # Пропускаем это вхождение как упоминание автомобиля
                    
                    is_valid = True
                    break
                
                if not is_valid:
                    continue  # Пропускаем тикер X5, если все вхождения - это автомобили
            
            # Для коротких тикеров (1-2 символа) проверяем контекст в исходном тексте,
            # чтобы не находить "T" в "T+1", "T-1" или "X" в "X5"
            if len(ticker) <= 2:
                ticker_lower = ticker.lower()
                # Ищем все вхождения тикера в исходном тексте
                pattern = rf"\b{re.escape(ticker_lower)}\b"
                matches = list(re.finditer(pattern, text_lower))
                
                # Проверяем каждое вхождение
                is_valid = False
                for match in matches:
                    start, end = match.span()
                    # Проверяем, не является ли это частью выражения "T+1", "T-1", "T+", "T-"
                    # или частью URL "t.me"
                    before = text_lower[max(0, start-3):start]
                    after = text_lower[end:min(len(text_lower), end+3)]
                    
                    # Пропускаем, если это часть "T+1", "T-1", "T+", "T-"
                    if re.search(r'[+\-]\s*\d+', after) or re.search(r'[+\-]\s*$', after):
                        continue
                    # Пропускаем, если это часть "t.me"
                    if after.startswith('.me') or before.endswith('t.'):
                        continue
                    # Пропускаем, если это часть "X5" (тикер X не должен находиться в "X5")
                    if after.startswith('5') and not before:
                        continue
                    
                    is_valid = True
                    break
                
                if not is_valid:
                    continue  # Пропускаем этот тикер
            
            found.add(ticker)

    # 2) поиск по паттернам (имена компаний)
    # Для коротких тикеров (1-2 символа) полностью пропускаем поиск по паттернам,
    # чтобы избежать ложных срабатываний на "T+1", "t.me", "X5 Group" и т.д.
    # Исключение: тикер T может искаться по паттернам (Т-банк, Т-Технологии)
    for ticker, info in companies.items():
        # Для коротких тикеров ищем только явные упоминания (уже сделано на этапе 1)
        # Исключение для T - разрешаем поиск по паттернам
        if len(ticker) <= 2 and ticker != "T":
            continue
        
        for p in info["patterns"]:
            # Пропускаем паттерны из самого тикера (они уже обработаны на этапе 1)
            if p == ticker.lower():
                continue
            
            pattern = rf"\b{re.escape(p)}\b"
            matches = list(re.finditer(pattern, text_norm))
            
            if matches:
                # Специальная проверка для тикера LENT - исключаем "лента/ленте новостей"
                if ticker == "LENT":
                    is_valid_mention = False
                    for match in matches:
                        start, end = match.span()
                        # Проверяем контекст после "лента/ленте" (до 30 символов)
                        after = text_norm[end:min(len(text_norm), end+30)]
                        # Проверяем контекст перед "лента/ленте" (до 20 символов)
                        before = text_norm[max(0, start-20):start]
                        
                        # Слова, указывающие на новостную ленту (не компанию)
                        news_feed_indicators = ['новостей', 'новостная', 'новостной', 'информации', 
                                                'информационная', 'сообщений', 'публикаций',
                                                'ранее опубликованной', 'опубликованной в']
                        
                        # Проверяем, не является ли это новостной лентой
                        is_news_feed = False
                        for indicator in news_feed_indicators:
                            # Проверяем первые 15 символов после "лента" для точного совпадения
                            # (чтобы не находить "новостей" через другие слова, как в "лента и лента новостей")
                            if indicator in after[:15]:
                                is_news_feed = True
                                break
                        
                        # Также проверяем, если перед "лента" стоит "новостная" или "информационная"
                        if 'новостная' in before[-15:] or 'информационная' in before[-15:]:
                            is_news_feed = True
                        
                        # Если это не новостная лента, то это валидное упоминание компании
                        if not is_news_feed:
                            is_valid_mention = True
                            break
                    
                    if not is_valid_mention:
                        continue  # Пропускаем тикер LENT, если все вхождения - это новостная лента
                
                found.add(ticker)
                break

    return sorted(list(found))  # сортируем для консистентности


def add_tickers_column_to_table(conn, table_name):
    """Добавляет колонку tickers в таблицу, если её нет"""
    cursor = conn.cursor()
    try:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN tickers TEXT")
        conn.commit()
        print(f"   Добавлена колонка tickers в таблицу {table_name}")
        return True
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
            print(f"   Колонка tickers уже существует в таблице {table_name}")
            return False
        else:
            raise


def process_table(conn, table_name, companies):
    """Обрабатывает одну таблицу: находит тикеры и обновляет колонку"""
    cursor = conn.cursor()
    
    # Получаем все сообщения
    cursor.execute(f"SELECT message_id, message FROM {table_name} WHERE message IS NOT NULL AND message != ''")
    rows = cursor.fetchall()
    
    print(f"   Найдено сообщений: {len(rows)}")
    
    updated = 0
    for message_id, message in rows:
        tickers = find_tickers(message, companies)
        
        if tickers:
            # Сохраняем как JSON строку (для множественных тикеров)
            tickers_json = json.dumps(tickers, ensure_ascii=False)
            cursor.execute(
                f"UPDATE {table_name} SET tickers = ? WHERE message_id = ?",
                (tickers_json, message_id)
            )
            updated += 1
    
    conn.commit()
    print(f"   Обновлено сообщений с тикерами: {updated}")
    return updated


def main():
    """Основная функция"""
    print("=" * 60)
    print("ПОИСК ТИКЕРОВ В НОВОСТЯХ")
    print("=" * 60)
    
    # 1. Загружаем тикеры из JSON
    print(f"\nЗагрузка тикеров из {TICKERS_JSON}...")
    if not TICKERS_JSON.exists():
        print(f"Файл не найден: {TICKERS_JSON}")
        return
    
    company_rows = load_tickers_from_json(TICKERS_JSON)
    companies = build_companies_dict(company_rows)
    print(f"Загружено компаний: {len(companies)}")
    
    # 2. Подключаемся к БД
    print(f"\nПодключение к базе данных {DB_PATH}...")
    if not DB_PATH.exists():
        print(f"База данных не найдена: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 3. Находим все таблицы с новостями (исключаем служебные)
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name != 'message_embeddings' "
        "AND name != 'duplicates'"
    )
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Найдено таблиц: {len(tables)}")
    print(f"  Таблицы: {', '.join(tables)}")
    
    # 4. Обрабатываем каждую таблицу
    total_updated = 0
    for table in tables:
        print(f"\n{'=' * 60}")
        print(f"Обработка таблицы: {table}")
        print(f"{'=' * 60}")
        
        # Добавляем колонку tickers, если её нет
        add_tickers_column_to_table(conn, table)
        
        # Обрабатываем сообщения
        updated = process_table(conn, table, companies)
        total_updated += updated
    
    conn.close()
    
    print(f"\n{'=' * 60}")
    print(f"ГОТОВО!")
    print(f"   Всего обновлено сообщений: {total_updated}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()