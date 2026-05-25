"""
Асинхронный парсер новостей с сайта Lenta.ru.
Поддерживает парсинг по категориям за указанный период.
"""
import asyncio
import configparser
import json
import logging
import re
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

from db_handler import NewsDatabase

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Заголовки для запросов
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


class LentaParser:
    """Асинхронный парсер новостей Lenta.ru."""

    def __init__(self, config_path: str = "config.ini"):
        """
        Инициализация парсера.

        Args:
            config_path: Путь к файлу конфигурации
        """
        # Отключаем интерполяцию, чтобы % в форматах дат не вызывали ошибки
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_path, encoding="utf-8")

        self.base_url = self.config.get("lenta", "base_url", fallback="https://lenta.ru")
        self.categories = [
            cat.strip()
            for cat in self.config.get("lenta", "categories", fallback="world").split(",")
        ]
        self.date_format = self.config.get("lenta", "date_format", fallback="%Y/%m/%d")
        self.request_delay = self.config.getfloat("parsers", "request_delay", fallback=0.5)
        self.concurrent_requests = self.config.getint("parsers", "concurrent_requests", fallback=10)
        self.request_timeout = self.config.getint("parsers", "request_timeout", fallback=30)
        self.retries = self.config.getint("parsers", "retries", fallback=3)
        self.verify_ssl = self.config.getboolean("parsers", "verify_ssl", fallback=False)

        db_path = self.config.get("database", "lenta_db_path", fallback="lenta_news.db")
        self.db = NewsDatabase(db_path)

        self.semaphore = asyncio.Semaphore(self.concurrent_requests)

    def get_parsing_dates(self) -> tuple[date, date]:
        """
        Получение дат начала и конца периода парсинга из конфига.

        Returns:
            Кортеж (start_date, end_date)
        """
        start_date_str = self.config.get("lenta", "start_date", fallback=None)
        end_date_str = self.config.get("lenta", "end_date", fallback=None)

        if start_date_str and end_date_str:
            try:
                start_date = date.fromisoformat(start_date_str)
                end_date = date.fromisoformat(end_date_str)
                return start_date, end_date
            except ValueError as e:
                logger.error(f"Ошибка парсинга дат из конфига: {e}")
                # Возвращаем значения по умолчанию
                end_date = date.today()
                start_date = end_date - timedelta(days=7)
                return start_date, end_date
        else:
            # Если даты не указаны, используем последнюю неделю
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            logger.warning("Даты не указаны в конфиге, используется последняя неделя")
            return start_date, end_date

    def _archive_url_for_date(self, d: date, category: Optional[str] = None) -> str:
        """
        Формирование URL архива на конкретный день.

        Args:
            d: Дата
            category: Категория (опционально)

        Returns:
            URL архива
        """
        date_str = d.strftime(self.date_format)
        if category:
            return f"{self.base_url}/rubrics/{category}/{date_str}/"
        return f"{self.base_url}/{date_str}/"

    async def _fetch_html(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[str]:
        """
        Асинхронная загрузка HTML с повторами.

        Args:
            session: Сессия aiohttp
            url: URL для загрузки

        Returns:
            HTML содержимое или None при ошибке
        """
        async with self.semaphore:
            for attempt in range(1, self.retries + 1):
                try:
                    async with session.get(
                        url, headers=HEADERS, timeout=self.request_timeout
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(
                                f"[{resp.status}] {url} (попытка {attempt}/{self.retries})"
                            )
                            if attempt < self.retries:
                                await asyncio.sleep(1)
                            continue

                        html = await resp.text()
                        await asyncio.sleep(self.request_delay)
                        return html

                except asyncio.TimeoutError:
                    logger.warning(f"Таймаут {url} (попытка {attempt}/{self.retries})")
                    if attempt < self.retries:
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {url}: {e} (попытка {attempt}/{self.retries})")
                    if attempt < self.retries:
                        await asyncio.sleep(1)

            return None

    async def _extract_article_links(
        self, session: aiohttp.ClientSession, d: date, category: Optional[str] = None
    ) -> list[str]:
        """
        Извлечение ссылок на статьи из архива за день.

        Args:
            session: Сессия aiohttp
            d: Дата
            category: Категория (опционально)

        Returns:
            Список URL статей
        """
        url = self._archive_url_for_date(d, category)
        html = await self._fetch_html(session, url)

        if html is None:
            logger.warning(f"Не удалось загрузить архив за {d} (категория: {category})")
            return []

        soup = BeautifulSoup(html, "html.parser")
        links: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Новости имеют путь /news/ГОД/МЕСЯЦ/ДЕНЬ/...
            if href.startswith("/news/"):
                full_url = urljoin(self.base_url, href.split("?")[0])
                links.add(full_url)

        links_list = sorted(links)
        logger.info(
            f"{d} (категория: {category or 'все'}): найдено ссылок {len(links_list)}"
        )
        return links_list

    async def _parse_article(
        self, session: aiohttp.ClientSession, url: str, fallback_date: Optional[date] = None
    ) -> Optional[dict]:
        """
        Парсинг одной статьи.

        Args:
            session: Сессия aiohttp
            url: URL статьи
            fallback_date: Дата архива для использования, если дата не найдена в статье

        Returns:
            Словарь с данными статьи или None
        """
        html = await self._fetch_html(session, url)
        if html is None:
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Заголовок
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Извлечение текста статьи
        paragraphs = []
        candidate_selectors = [
            'div[itemprop="articleBody"] p',
            "div.js-topic__text p",
            "div.topic-body__content p",
            "article p",
        ]

        for sel in candidate_selectors:
            blocks = soup.select(sel)
            if blocks:
                paragraphs = blocks
                break

        if not paragraphs:
            paragraphs = soup.find_all("p")

        text_parts: list[str] = []
        for p in paragraphs:
            t = p.get_text(" ", strip=True)
            if not t:
                continue
            # Обрезка типичных хвостов
            if "Материалы по теме" in t or "Что думаешь? Оцени!" in t:
                break
            text_parts.append(t)

        text = "\n".join(text_parts).strip()

        if not text or not title:
            return None

        # Попытка извлечь дату публикации (с fallback датой архива)
        publication_datetime = self._extract_publication_datetime(soup, url, fallback_date)

        # Попытка извлечь теги
        tags = self._extract_tags(soup)

        return {
            "url": url,
            "title": title,
            "text": text,
            "publication_datetime": publication_datetime,
            "tags": tags,
        }

    def _extract_date_from_url(self, url: str) -> Optional[datetime]:
        """
        Извлечение даты из URL статьи.
        Формат URL: /news/2025/10/15/... или /news/2025/10/15/...

        Args:
            url: URL статьи

        Returns:
            datetime объект или None
        """
        # Ищем паттерн /news/YYYY/MM/DD/ в URL
        match = re.search(r'/news/(\d{4})/(\d{2})/(\d{2})/', url)
        if match:
            try:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day, 12, 0, 0)  # Полдень по умолчанию
            except ValueError:
                pass
        return None

    def _extract_publication_datetime(
        self, soup: BeautifulSoup, url: str, fallback_date: Optional[date] = None
    ) -> datetime:
        """
        Извлечение даты и времени публикации из статьи.

        Args:
            soup: BeautifulSoup объект
            url: URL статьи (для извлечения даты из URL)
            fallback_date: Дата архива как резервный вариант

        Returns:
            datetime объект
        """
        # Попытка найти JSON-LD структурированные данные
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Ищем datePublished в различных местах
                    date_published = (
                        data.get('datePublished')
                        or data.get('dateCreated')
                        or (data.get('article', {}) if isinstance(data.get('article'), dict) else {}).get('datePublished')
                    )
                    if date_published:
                        try:
                            # Пробуем различные форматы
                            for fmt in [
                                "%Y-%m-%dT%H:%M:%S",
                                "%Y-%m-%dT%H:%M:%S%z",
                                "%Y-%m-%dT%H:%M:%S.%f",
                                "%Y-%m-%dT%H:%M:%S.%f%z",
                            ]:
                                try:
                                    # Убираем таймзону для парсинга
                                    date_str = date_published.split('+')[0].split('-')[0] if '+' in date_published or '-' in date_published[-6:] else date_published
                                    parsed = datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
                                    logger.debug(f"Дата и время извлечены из JSON-LD: {parsed}")
                                    return parsed
                                except (ValueError, IndexError):
                                    continue
                        except Exception as e:
                            logger.debug(f"Ошибка при парсинге JSON-LD даты: {e}")
            except (json.JSONDecodeError, AttributeError):
                continue

        # Сначала ищем специфичный элемент Lenta.ru с форматом "09:06, 15 октября 2025" или "09:06, 15.10.2025"
        topic_header_time = soup.select_one('a.topic-header__time, a[class*="topic-header__time"]')
        if topic_header_time:
            time_text = topic_header_time.get_text(strip=True)
            # Формат с русским месяцем: "09:06, 15 октября 2025"
            match_ru = re.match(r'(\d{1,2}):(\d{2}),\s*(\d{1,2})\s+(\w+)\s+(\d{4})', time_text)
            if match_ru:
                try:
                    hour, minute, day, month_name, year = match_ru.groups()
                    # Словарь русских названий месяцев
                    months_ru = {
                        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
                        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
                        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
                    }
                    month = months_ru.get(month_name.lower())
                    if month:
                        result = datetime(int(year), month, int(day), int(hour), int(minute), 0)
                        logger.debug(f"Дата и время извлечены из topic-header__time (русский формат): {result}")
                        return result
                except (ValueError, KeyError) as e:
                    logger.debug(f"Ошибка при парсинге topic-header__time '{time_text}': {e}")
            
            # Формат с числовым месяцем: "09:06, 15.10.2025"
            match_num = re.match(r'(\d{1,2}):(\d{2}),\s*(\d{1,2})\.(\d{1,2})\.(\d{4})', time_text)
            if match_num:
                try:
                    hour, minute, day, month, year = match_num.groups()
                    result = datetime(int(year), int(month), int(day), int(hour), int(minute), 0)
                    logger.debug(f"Дата и время извлечены из topic-header__time (числовой формат): {result}")
                    return result
                except ValueError as e:
                    logger.debug(f"Ошибка при парсинге topic-header__time '{time_text}': {e}")

        # Расширенные селекторы для поиска даты и времени (специфичные для Lenta.ru)
        time_selectors = [
            'time[itemprop="datePublished"]',
            'time[datetime]',
            'time[pubdate]',
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]',
            'meta[name="pubdate"]',
            'meta[property="og:published_time"]',
            'meta[name="article:published_time"]',
            'span[class*="time"]',
            'span[class*="date"]',
            'div[class*="time"]',
            'div[class*="date"]',
            'div[class*="published"]',
            'span[class*="published"]',
            'div[class*="topic-header__time"]',  # Специфично для Lenta.ru
            'span[class*="topic-header__time"]',
            'time[class*="topic-header__time"]',
            'a[class*="topic-header__time"]',  # Ссылка с временем
        ]

        # Попытка найти полную дату и время в HTML
        for sel in time_selectors:
            time_tag = soup.select_one(sel)
            if time_tag:
                # Сначала пробуем атрибут datetime или content
                datetime_str = (
                    time_tag.get("datetime") 
                    or time_tag.get("content")
                    or time_tag.get_text(strip=True)
                )
                if datetime_str:
                    # Очистка строки от лишних символов
                    datetime_str = datetime_str.strip()
                    try:
                        # Расширенный список форматов с временем
                        for fmt in [
                            "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%dT%H:%M:%S%z",
                            "%Y-%m-%dT%H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%S.%f%z",
                            "%Y-%m-%d %H:%M:%S",
                            "%d.%m.%Y %H:%M:%S",
                            "%d.%m.%Y, %H:%M",  # Формат Lenta.ru: "15.10.2025, 14:30"
                            "%d.%m.%Y %H:%M",
                            "%Y-%m-%d %H:%M",
                            "%d/%m/%Y %H:%M:%S",
                            "%d/%m/%Y %H:%M",
                            "%Y-%m-%d",
                            "%d.%m.%Y",
                        ]:
                            try:
                                # Пробуем распарсить
                                parsed = datetime.strptime(datetime_str[:len(fmt)+10], fmt)
                                logger.debug(f"Дата и время извлечены из HTML ({sel}): {parsed}")
                                return parsed
                            except (ValueError, IndexError):
                                continue
                    except Exception as e:
                        logger.debug(f"Ошибка при парсинге даты '{datetime_str}': {e}")
                        pass

        # Попытка найти дату и время вместе в тексте (формат Lenta.ru: "15.10.2025, 14:30")
        # Ищем паттерн: дата, время или дата время
        datetime_patterns = [
            r'(\d{1,2})\.(\d{1,2})\.(\d{4}),\s*(\d{1,2}):(\d{2})',  # "15.10.2025, 14:30"
            r'(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})',   # "15.10.2025 14:30"
            r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2}):(\d{2})', # "2025-10-15 14:30:00"
            r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})',         # "2025-10-15 14:30"
        ]
        
        # Ищем в тексте страницы (но только в начале, где обычно находится дата публикации)
        page_text = soup.get_text()
        # Берем первые 2000 символов, где обычно находится дата публикации
        header_text = page_text[:2000]
        
        for pattern in datetime_patterns:
            match = re.search(pattern, header_text)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 5:  # Формат с точками
                        day, month, year, hour, minute = map(int, groups)
                        result = datetime(year, month, day, hour, minute, 0)
                        logger.debug(f"Дата и время извлечены из текста (формат 1): {result}")
                        return result
                    elif len(groups) == 6:  # Формат с дефисами и секундами
                        year, month, day, hour, minute, second = map(int, groups)
                        result = datetime(year, month, day, hour, minute, second)
                        logger.debug(f"Дата и время извлечены из текста (формат 2): {result}")
                        return result
                    elif len(groups) == 5 and '-' in pattern:  # Формат с дефисами без секунд
                        year, month, day, hour, minute = map(int, groups)
                        result = datetime(year, month, day, hour, minute, 0)
                        logger.debug(f"Дата и время извлечены из текста (формат 3): {result}")
                        return result
                except (ValueError, IndexError) as e:
                    logger.debug(f"Ошибка при парсинге паттерна {pattern}: {e}")
                    continue
        
        # Попытка найти время отдельно, если дата уже известна
        found_date = None
        found_time = None
        
        # Ищем дату в тексте
        date_patterns = [
            r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
            r'(\d{4})-(\d{2})-(\d{2})',
        ]
        
        time_patterns = [
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, header_text)
            if match:
                try:
                    if '.' in pattern:
                        day, month, year = map(int, match.groups())
                    else:
                        year, month, day = map(int, match.groups())
                    found_date = date(year, month, day)
                    break
                except ValueError:
                    continue
        
        # Ищем время рядом с датой (в пределах 50 символов)
        if found_date:
            date_pos = header_text.find(str(found_date.day))
            if date_pos != -1:
                # Ищем время в окне ±50 символов от даты
                search_window = header_text[max(0, date_pos-50):date_pos+200]
                for pattern in time_patterns:
                    match = re.search(pattern, search_window)
                    if match:
                        try:
                            hour, minute = map(int, match.groups()[:2])
                            second = int(match.group(3)) if match.group(3) else 0
                            found_time = (hour, minute, second)
                            break
                        except (ValueError, IndexError):
                            continue
        
        # Если нашли и дату, и время
        if found_date and found_time:
            result = datetime(found_date.year, found_date.month, found_date.day, 
                            found_time[0], found_time[1], found_time[2])
            logger.debug(f"Дата и время извлечены из текста (раздельно): {result}")
            return result
        
        # Если нашли только дату, используем её с временем из URL или fallback
        if found_date:
            # Пробуем извлечь время из URL (если есть)
            url_time_match = re.search(r'/(\d{2}):(\d{2}):(\d{2})/', url)
            if url_time_match:
                try:
                    hour, minute, second = map(int, url_time_match.groups())
                    result = datetime(found_date.year, found_date.month, found_date.day, hour, minute, second)
                    logger.debug(f"Дата из текста, время из URL: {result}")
                    return result
                except ValueError:
                    pass
            # Используем полдень как разумное значение по умолчанию
            result = datetime.combine(found_date, datetime.min.time().replace(hour=12))
            logger.debug(f"Дата из текста, время по умолчанию: {result}")
            return result

        # Попытка извлечь дату из URL
        url_date = self._extract_date_from_url(url)
        if url_date:
            logger.debug(f"Дата извлечена из URL: {url_date}")
            return url_date

        # Используем дату архива как fallback (с полднем)
        if fallback_date:
            fallback_datetime = datetime.combine(fallback_date, datetime.min.time().replace(hour=12))
            logger.debug(f"Использована дата архива: {fallback_datetime}")
            return fallback_datetime

        # Последний вариант - текущее время (не должно доходить до этого)
        logger.warning(f"Не удалось извлечь дату для {url}, используется текущее время")
        return datetime.now()

    def _extract_tags(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Извлечение тегов из статьи.

        Args:
            soup: BeautifulSoup объект

        Returns:
            Строка с тегами через запятую или None
        """
        tag_selectors = [
            'a[class*="tag"]',
            'span[class*="tag"]',
            'div[class*="tag"]',
        ]

        tags = []
        for sel in tag_selectors:
            tag_elements = soup.select(sel)
            for tag_elem in tag_elements:
                tag_text = tag_elem.get_text(strip=True)
                if tag_text and len(tag_text) < 50:  # Фильтр слишком длинных строк
                    tags.append(tag_text)

        return ", ".join(tags) if tags else None

    async def _process_article(
        self,
        session: aiohttp.ClientSession,
        url: str,
        archive_date: date,
        category: Optional[str] = None,
    ) -> None:
        """
        Обработка одной статьи: парсинг и сохранение в БД.

        Args:
            session: Сессия aiohttp
            url: URL статьи
            archive_date: Дата архива (используется как fallback для даты публикации)
            category: Категория (опционально)
        """
        # Проверка на существование в БД (с указанием категории)
        if category and self.db.news_exists(url, category):
            logger.debug(f"Пропущена существующая новость: {url} (категория: {category})")
            return

        article_data = await self._parse_article(session, url, archive_date)
        if not article_data:
            logger.warning(f"Не удалось распарсить статью: {url}")
            return

        # Категория обязательна для сохранения
        if not category:
            logger.warning(f"Категория не указана, новость не будет сохранена: {url}")
            return

        # Сохранение в БД
        self.db.save_news(
            url=article_data["url"],
            title=article_data["title"],
            text=article_data["text"],
            publication_datetime=article_data["publication_datetime"],
            tags=article_data.get("tags"),
            category=category,
        )

    async def _crawl_date(
        self, session: aiohttp.ClientSession, d: date, category: Optional[str] = None
    ) -> None:
        """
        Парсинг всех статей за один день.

        Args:
            session: Сессия aiohttp
            d: Дата
            category: Категория (опционально)
        """
        logger.info(f"=== Парсинг {d} (категория: {category or 'все'}) ===")

        links = await self._extract_article_links(session, d, category)
        if not links:
            return

        # Обработка статей с ограничением параллелизма
        tasks = [
            self._process_article(session, link, d, category) for link in links
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def crawl(
        self,
        start_date: date,
        end_date: date,
        categories: Optional[list[str]] = None,
    ) -> None:
        """
        Основной метод парсинга за период.

        Args:
            start_date: Дата начала
            end_date: Дата окончания
            categories: Список категорий для парсинга (если None, используются из конфига)
        """
        if categories is None:
            categories = self.categories

        logger.info(
            f"Начало парсинга Lenta.ru с {start_date} по {end_date}, "
            f"категории: {categories}"
        )

        # Настройка SSL
        ssl_context = None if self.verify_ssl else False
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            d = start_date
            while d <= end_date:
                # Парсинг по категориям
                for category in categories:
                    await self._crawl_date(session, d, category)
                    # Небольшая пауза между категориями
                    await asyncio.sleep(self.request_delay)

                d += timedelta(days=1)

        logger.info("Парсинг завершен")

    def close(self) -> None:
        """Закрытие подключения к БД."""
        self.db.close()


async def main():
    """Пример использования парсера."""
    parser = LentaParser()
    try:
        # Получаем даты из конфига
        start_date, end_date = parser.get_parsing_dates()
        logger.info(f"Период парсинга: с {start_date} по {end_date}")
        await parser.crawl(start_date, end_date)
    finally:
        parser.close()


if __name__ == "__main__":
    asyncio.run(main())

