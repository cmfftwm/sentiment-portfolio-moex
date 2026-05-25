"""
Модуль для работы с базой данных новостей.
Поддерживает SQLite (по умолчанию) и может быть расширен для других БД.
"""
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NewsDatabase:
    """Класс для работы с базой данных новостей."""

    def __init__(self, db_path: str):
        """
        Инициализация подключения к БД.

        Args:
            db_path: Путь к файлу БД SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row

    def _sanitize_table_name(self, category: str) -> str:
        """
        Очистка имени категории для использования в качестве имени таблицы.

        Args:
            category: Название категории

        Returns:
            Безопасное имя таблицы
        """
        import re
        # Словарь соответствий для известных категорий
        category_map = {
            'Экономика': 'economica',
            'Политика': 'politika',
            'Лента новостей': 'lenta_novostey',
        }
        
        # Проверяем, есть ли прямое соответствие
        if category in category_map:
            return category_map[category]
        
        # Если нет прямого соответствия, используем транслитерацию
        translit_map = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch',
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
        }
        
        # Транслитерируем кириллицу
        safe_name = ''.join(translit_map.get(c, c) for c in category)
        # Заменяем оставшиеся недопустимые символы на подчеркивания
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', safe_name)
        # Убираем множественные подчеркивания
        safe_name = re.sub(r'_+', '_', safe_name)
        # Убираем подчеркивания в начале и конце
        safe_name = safe_name.strip('_')
        return safe_name if safe_name else 'default'

    def _create_table_for_category(self, category: str) -> str:
        """
        Создание таблицы для конкретной категории, если её нет.

        Args:
            category: Название категории

        Returns:
            Имя созданной таблицы
        """
        table_name = self._sanitize_table_name(category)
        cursor = self.conn.cursor()

        # Создание таблицы
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                news_id TEXT PRIMARY KEY,
                publication_datetime TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                title TEXT NOT NULL,
                tags TEXT
            )
            """
        )

        # Создание индексов
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_url ON "{table_name}"(url);
            """
        )
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_publication_datetime 
            ON "{table_name}"(publication_datetime);
            """
        )

        self.conn.commit()
        logger.debug(f"Таблица '{table_name}' создана/проверена для категории '{category}'")
        return table_name

    def _generate_news_id(self, url: str) -> str:
        """
        Генерация уникального ID новости на основе URL.

        Args:
            url: URL новости

        Returns:
            Хеш URL в виде строки
        """
        return hashlib.md5(url.encode()).hexdigest()

    def save_news(
        self,
        url: str,
        title: str,
        text: str,
        publication_datetime: datetime,
        tags: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        """
        Сохранение новости в БД в таблицу соответствующей категории.

        Args:
            url: URL новости
            title: Заголовок
            text: Текст новости
            publication_datetime: Дата и время публикации
            tags: Теги (опционально)
            category: Категория (обязательно для определения таблицы)

        Returns:
            True если новость сохранена, False если уже существует
        """
        if not category:
            logger.warning("Категория не указана, новость не будет сохранена")
            return False

        table_name = self._create_table_for_category(category)
        news_id = self._generate_news_id(url)
        pub_datetime_str = publication_datetime.isoformat()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"""
                INSERT OR IGNORE INTO "{table_name}" 
                (news_id, publication_datetime, url, text, title, tags)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    news_id,
                    pub_datetime_str,
                    url,
                    text,
                    title,
                    tags,
                ),
            )
            self.conn.commit()

            if cursor.rowcount > 0:
                logger.debug(f"Сохранена новость в таблицу '{table_name}': {url}")
                return True
            else:
                logger.debug(f"Новость уже существует в таблице '{table_name}': {url}")
                return False

        except sqlite3.Error as e:
            logger.error(f"Ошибка при сохранении новости {url} в таблицу '{table_name}': {e}")
            self.conn.rollback()
            return False

    def news_exists(self, url: str, category: Optional[str] = None) -> bool:
        """
        Проверка существования новости в БД.

        Args:
            url: URL новости
            category: Категория для проверки в конкретной таблице (если None, проверяет во всех)

        Returns:
            True если новость существует
        """
        cursor = self.conn.cursor()

        if category:
            # Проверка в конкретной таблице категории
            table_name = self._sanitize_table_name(category)
            try:
                cursor.execute(f'SELECT 1 FROM "{table_name}" WHERE url = ? LIMIT 1', (url,))
                return cursor.fetchone() is not None
            except sqlite3.OperationalError:
                # Таблица не существует
                return False
        else:
            # Проверка во всех таблицах (получаем список всех таблиц)
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f'SELECT 1 FROM "{table}" WHERE url = ? LIMIT 1', (url,))
                if cursor.fetchone():
                    return True
            return False

    def close(self) -> None:
        """Закрытие подключения к БД."""
        if self.conn:
            self.conn.close()
            logger.info(f"Подключение к БД закрыто: {self.db_path}")

    def __enter__(self):
        """Контекстный менеджер: вход."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход."""
        self.close()

