"""
Асинхронный парсер новостей с сайта RIA.ru.
Поддерживает парсинг новостей через RSS и поиск по ключевым словам.
"""
import asyncio
import configparser
import logging
import re
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from dateutil.parser import parse

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


class RIAParser:
    """Асинхронный парсер новостей RIA.ru."""

    def __init__(self, config_path: str = "config.ini"):
        """
        Инициализация парсера.

        Args:
            config_path: Путь к файлу конфигурации
        """
        # Отключаем интерполяцию, чтобы % в форматах дат не вызывали ошибки
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_path, encoding="utf-8")

        self.base_url = self.config.get(
            "ria",
            "base_url",
            fallback="https://ria.ru",
        )
        self.search_url = self.config.get(
            "ria",
            "search_url",
            fallback="https://ria.ru/search/",
        )
        self.rss_url = self.config.get(
            "ria",
            "rss_url",
            fallback="https://ria.ru/export/rss2/archive/index.xml",
        )
        
        # Читаем категории из конфига
        categories_str = self.config.get("ria", "categories", fallback="")
        if categories_str:
            self.categories = [cat.strip() for cat in categories_str.split(",") if cat.strip()]
        else:
            self.categories = []
        
        # Словарь URL для категорий (прямые ссылки на разделы)
        self.category_urls = {
            "Экономика": "https://ria.ru/economy/",
            "Политика": "https://ria.ru/politics/",
        }
        
        self.date_format = self.config.get(
            "ria",
            "date_format",
            fallback="%Y%m%d",
        )

        self.request_delay = self.config.getfloat("parsers", "request_delay", fallback=0.5)
        self.concurrent_requests = self.config.getint("parsers", "concurrent_requests", fallback=10)
        self.request_timeout = self.config.getint("parsers", "request_timeout", fallback=30)
        self.retries = self.config.getint("parsers", "retries", fallback=3)
        self.verify_ssl = self.config.getboolean("parsers", "verify_ssl", fallback=False)

        db_path = self.config.get("database", "ria_db_path", fallback="ria_news.db")
        self.db = NewsDatabase(db_path)

        self.semaphore = asyncio.Semaphore(self.concurrent_requests)

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """
        Парсинг даты из строки.

        Args:
            date_str: Строка с датой

        Returns:
            datetime объект в UTC
        """
        dt = parse(date_str)
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            # Если дата наивная (без часового пояса), считаем ее UTC
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Если дата с часовым поясом, конвертируем ее в UTC
            return dt.astimezone(timezone.utc)

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
                                # Для 429 (Too Many Requests) используем экспоненциальную задержку
                                if resp.status == 429:
                                    # Экспоненциальная задержка: 5, 10, 20, 40, 80 секунд
                                    delay = min(5 * (2 ** (attempt - 1)), 80)
                                    logger.info(f"Ожидание {delay} секунд перед повторной попыткой...")
                                    await asyncio.sleep(delay)
                                else:
                                    # Для других ошибок - обычная задержка
                                    await asyncio.sleep(2)
                            continue

                        html = await resp.text()
                        await asyncio.sleep(self.request_delay)
                        return html

                except asyncio.TimeoutError:
                    logger.warning(f"Таймаут {url} (попытка {attempt}/{self.retries})")
                    if attempt < self.retries:
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.error(
                        f"Ошибка при загрузке {url}: {e} (попытка {attempt}/{self.retries})"
                    )
                    if attempt < self.retries:
                        await asyncio.sleep(2)

            return None

    def get_parsing_dates(self) -> tuple[date, date]:
        """
        Получение дат начала и конца периода парсинга из конфига.

        Returns:
            Кортеж (start_date, end_date)
        """
        start_date_str = self.config.get("ria", "start_date", fallback=None)
        end_date_str = self.config.get("ria", "end_date", fallback=None)

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

    def _get_last_news_datetime(self) -> Optional[datetime]:
        """
        Получение даты последней новости из БД по всем категориям.

        Returns:
            datetime последней новости или None
        """
        try:
            cursor = self.db.conn.cursor()
            
            # Получаем список всех таблиц (категорий)
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                return None
            
            # Находим самую последнюю дату среди всех категорий
            latest_datetime = None
            for table_name in tables:
                try:
                    cursor.execute(
                        f'SELECT publication_datetime FROM "{table_name}" '
                        f'ORDER BY publication_datetime DESC LIMIT 1'
                    )
                    row = cursor.fetchone()
                    if row:
                        dt_str = row[0]
                        dt = datetime.fromisoformat(dt_str)
                        if latest_datetime is None or dt > latest_datetime:
                            latest_datetime = dt
                except Exception as e:
                    logger.debug(f"Ошибка при проверке таблицы '{table_name}': {e}")
                    continue
            
            return latest_datetime
        except Exception as e:
            logger.error(f"Ошибка при получении последней новости из БД: {e}")
            return None

    def _get_category_url_for_date(self, category: str, d: date) -> str:
        """
        Формирование URL категории с датой.

        Args:
            category: Категория
            d: Дата

        Returns:
            URL с датой
        """
        if category not in self.category_urls:
            return self.search_url
        
        base_url = self.category_urls[category]
        # Формат даты в URL RIA: /YYYYMMDD/
        date_str = d.strftime("%Y%m%d")
        return f"{base_url}{date_str}/"
    
    async def fetch_links(
        self, session: aiohttp.ClientSession, category: Optional[str] = None, 
        start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Получение списка ссылок на новости со страницы категории.

        Args:
            session: Сессия aiohttp
            category: Категория для парсинга (опционально)
            start_date: Дата начала периода (опционально)
            end_date: Дата окончания периода (опционально)

        Returns:
            Список словарей с данными о новостях
        """
        all_news_data = []
        
        # Если указан период, парсим по дням
        if start_date and end_date and category:
            current_date = start_date
            while current_date <= end_date:
                category_url = self._get_category_url_for_date(category, current_date)
                logger.info(f"Парсинг {category} за {current_date}: {category_url}")
                
                response = await self._fetch_html(session, category_url)
                if response:
                    news_data = self._parse_news_from_page(response, category, current_date)
                    logger.info(f"Найдено {len(news_data)} новостей для {category} за {current_date}")
                    all_news_data.extend(news_data)
                else:
                    logger.warning(f"Не удалось получить ответ для {category} за {current_date} (URL: {category_url})")
                
                current_date += timedelta(days=1)
        else:
            # Если период не указан, парсим основную страницу категории
            if category and category in self.category_urls:
                search_url = self.category_urls[category]
            else:
                search_url = self.search_url

            response = await self._fetch_html(session, search_url)
            if response:
                all_news_data = self._parse_news_from_page(response, category)
        
        return all_news_data
    
    def _parse_news_from_page(
        self, html: str, category: Optional[str] = None, fallback_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Парсинг новостей из HTML страницы.

        Args:
            html: HTML содержимое страницы
            category: Категория (опционально)
            fallback_date: Дата для использования, если не найдена в новости (опционально)

        Returns:
            Список словарей с данными о новостях
        """
        soup = BeautifulSoup(html, "html.parser")
        news_data = []

        # Используем категорию из параметра
        used_category = category or (self.categories[0] if self.categories else "Лента новостей")

        # Парсим новости напрямую со страницы
        news_items = soup.find_all("div", {"class": "list-item", "data-type": "article"})
        logger.debug(f"Найдено {len(news_items)} элементов с классом 'list-item' для категории '{used_category}'")
        
        # Если не нашли новости стандартным способом, пробуем альтернативные селекторы
        if not news_items:
            # Пробуем другие варианты селекторов
            news_items = soup.find_all("div", class_=lambda x: x and "list-item" in x)
            logger.debug(f"Альтернативный поиск: найдено {len(news_items)} элементов")
            
            if not news_items:
                # Пробуем найти любые ссылки на новости
                news_items = soup.find_all("a", href=re.compile(r'/20\d{6}/'))
                logger.debug(f"Поиск по ссылкам: найдено {len(news_items)} ссылок")
        
        for news_item in news_items:
            link_elem = news_item.find("a")
            if not link_elem or not link_elem.get("href"):
                continue

            link = link_elem.get("href")
            if not link.startswith("http"):
                link = self.base_url + link if link.startswith("/") else f"{self.base_url}/{link}"

            # Извлекаем заголовок - пробуем разные варианты
            title = ""
            # Сначала ищем в ссылке
            title_elem = news_item.find("a")
            if title_elem:
                title = title_elem.get_text(strip=True)
                # Если в ссылке нет текста, ищем в дочерних элементах
                if not title:
                    title_span = title_elem.find("span") or title_elem.find("div")
                    if title_span:
                        title = title_span.get_text(strip=True)
            
            # Если не нашли, пробуем другие варианты
            if not title:
                title_elem = news_item.find("h2") or news_item.find("h3") or news_item.find("h4")
                if title_elem:
                    title = title_elem.get_text(strip=True)
            
            # Если все еще не нашли, пробуем найти любой текстовый элемент
            if not title:
                # Ищем элемент с классом, содержащим "title" или "head"
                title_elem = news_item.find(class_=lambda x: x and ("title" in x.lower() or "head" in x.lower()))
                if title_elem:
                    title = title_elem.get_text(strip=True)
            
            # Если заголовок все еще пустой, используем первые слова из текста
            if not title:
                text_elem = news_item.get_text(strip=True)
                if text_elem:
                    # Берем первые 100 символов как заголовок
                    title = text_elem[:100].strip()
                    if len(text_elem) > 100:
                        title += "..."

            # Извлекаем дату из новости - пробуем разные варианты
            news_date = None
            date_elem = (
                news_item.find("time") 
                or news_item.find(class_=lambda x: x and ("date" in x.lower() or "time" in x.lower()))
                or news_item.find("span", class_=lambda x: x and ("date" in x.lower() or "time" in x.lower()))
            )
            
            if date_elem:
                date_str = date_elem.get("datetime") or date_elem.get("data-time") or date_elem.get_text(strip=True)
                if date_str:
                    try:
                        news_date = self._parse_date(date_str)
                    except Exception as e:
                        logger.debug(f"Не удалось распарсить дату '{date_str}': {e}")
                        news_date = None
            
            # Если дата не найдена, пробуем извлечь из URL или используем fallback_date
            if not news_date:
                # Пробуем извлечь дату из URL (формат: /YYYYMMDD/...)
                url_date_match = re.search(r'/(\d{8})/', link)
                if url_date_match:
                    try:
                        date_str = url_date_match.group(1)
                        news_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                    except:
                        pass
                
                # Если все еще не найдена, используем fallback_date или текущую дату
                if not news_date:
                    if fallback_date:
                        news_date = datetime.combine(fallback_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                    else:
                        news_date = datetime.now(timezone.utc)
                    logger.debug(f"Дата не найдена для {link}, используется {news_date.date()}")

            news_data.append(
                {
                    "link": link,
                    "title": title,
                    "category": used_category,
                    "datetime": news_date,
                }
            )

        # Если новостей мало, пробуем парсить дополнительные страницы
        # (можно добавить пагинацию позже)
        
        return news_data

    def _extract_publication_datetime(
        self, soup: BeautifulSoup, url: str, fallback_datetime: Optional[datetime] = None
    ) -> datetime:
        """
        Извлечение даты и времени публикации из статьи RIA.ru.

        Args:
            soup: BeautifulSoup объект страницы
            url: URL статьи
            fallback_datetime: Резервная дата/время, если не найдено

        Returns:
            datetime объект
        """
        import json
        
        # Попытка найти JSON-LD структурированные данные
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    date_published = (
                        data.get('datePublished')
                        or data.get('dateCreated')
                        or (data.get('article', {}) if isinstance(data.get('article'), dict) else {}).get('datePublished')
                    )
                    if date_published:
                        try:
                            return self._parse_date(date_published)
                        except:
                            pass
            except (json.JSONDecodeError, AttributeError):
                continue

        # Ищем мета-теги с датой
        meta_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]',
            'meta[name="pubdate"]',
            'meta[property="og:published_time"]',
            'meta[name="article:published_time"]',
            'meta[itemprop="datePublished"]',
        ]
        
        for selector in meta_selectors:
            meta_tag = soup.select_one(selector)
            if meta_tag:
                date_str = meta_tag.get("content") or meta_tag.get("value")
                if date_str:
                    try:
                        return self._parse_date(date_str)
                    except:
                        continue

        # Ищем time элементы
        time_selectors = [
            'time[itemprop="datePublished"]',
            'time[datetime]',
            'time[pubdate]',
            'time[class*="date"]',
            'time[class*="time"]',
        ]
        
        for selector in time_selectors:
            time_tag = soup.select_one(selector)
            if time_tag:
                date_str = time_tag.get("datetime") or time_tag.get_text(strip=True)
                if date_str:
                    try:
                        return self._parse_date(date_str)
                    except:
                        continue

        # Ищем элементы с классами, содержащими date или time
        date_elements = soup.find_all(class_=lambda x: x and ("date" in x.lower() or "time" in x.lower()))
        for elem in date_elements:
            date_str = elem.get("datetime") or elem.get("data-time") or elem.get_text(strip=True)
            if date_str:
                # Пробуем распарсить различные форматы
                try:
                    return self._parse_date(date_str)
                except:
                    # Пробуем найти паттерн даты и времени в тексте
                    # Формат RIA: "20 ноября, 15:18" или "20.11.2025, 15:18"
                    patterns = [
                        r'(\d{1,2})\s+(\w+)\s+(\d{4}),\s*(\d{1,2}):(\d{2})',  # "20 ноября 2025, 15:18"
                        r'(\d{1,2})\.(\d{1,2})\.(\d{4}),\s*(\d{1,2}):(\d{2})',  # "20.11.2025, 15:18"
                        r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})',        # "2025-11-20 15:18"
                    ]
                    
                    months_ru = {
                        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
                        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
                        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
                    }
                    
                    for pattern in patterns:
                        match = re.search(pattern, date_str)
                        if match:
                            try:
                                if '.' in pattern:
                                    day, month, year, hour, minute = map(int, match.groups())
                                elif '-' in pattern:
                                    year, month, day, hour, minute = map(int, match.groups())
                                else:
                                    day, month_name, year, hour, minute = match.groups()
                                    month = months_ru.get(month_name.lower())
                                    if not month:
                                        continue
                                
                                return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
                            except (ValueError, KeyError):
                                continue
                    continue

        # Пробуем извлечь дату из URL
        url_date_match = re.search(r'/(\d{8})/', url)
        if url_date_match:
            try:
                date_str = url_date_match.group(1)
                news_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                # Используем полдень как время по умолчанию
                return news_date.replace(hour=12, minute=0, second=0)
            except:
                pass

        # Используем fallback или текущее время
        if fallback_datetime:
            return fallback_datetime
        
        logger.warning(f"Не удалось извлечь дату для {url}, используется текущее время")
        return datetime.now(timezone.utc)

    async def parse_news_item(
        self, session: aiohttp.ClientSession, link: str, **kwargs
    ) -> Optional[Dict]:
        """
        Парсинг одной новости.

        Args:
            session: Сессия aiohttp
            link: URL новости
            **kwargs: Дополнительные параметры (title, category, datetime)

        Returns:
            Словарь с данными новости или None
        """
        response = await self._fetch_html(session, link)

        if not response:
            logger.warning(f"Ошибка при запросе страницы: {link}")
            return None

        soup = BeautifulSoup(response, "html.parser")

        # Извлекаем заголовок со страницы новости, если он не был передан
        title = kwargs.get("title", "")
        if not title:
            # Пробуем разные варианты извлечения заголовка
            title_elem = (
                soup.find("h1", class_=lambda x: x and "article" in x.lower())
                or soup.find("h1")
                or soup.find("h2", class_=lambda x: x and "title" in x.lower())
                or soup.find("div", class_=lambda x: x and "title" in x.lower())
            )
            if title_elem:
                title = title_elem.get_text(strip=True)

        article = soup.find(
            "div", {"class": "article__body js-mediator-article mia-analytics"}
        )
        if not article:
            logger.warning(f"Не найдено содержимое статьи: {link}")
            return None

        news_content = ""
        
        # Пробуем разные варианты извлечения текста
        # Вариант 1: div с классом article__text
        text_divs = article.find_all("div", class_="article__text")
        if text_divs:
            for div in text_divs:
                text = div.get_text(separator="\n", strip=True)
                if text:
                    news_content += text + "\n\n"
        
        # Вариант 2: если не нашли, пробуем все параграфы внутри article
        if not news_content.strip():
            paragraphs = article.find_all("p")
            if paragraphs:
                for p in paragraphs:
                    text = p.get_text(separator=" ", strip=True)
                    if text and len(text) > 20:  # Пропускаем короткие элементы
                        news_content += text + "\n\n"
        
        # Вариант 3: если все еще пусто, берем весь текст из article
        if not news_content.strip():
            # Удаляем скрипты и стили
            for script in article(["script", "style", "noscript"]):
                script.decompose()
            news_content = article.get_text(separator="\n", strip=True)
            # Очищаем от множественных пустых строк
            news_content = "\n".join(line.strip() for line in news_content.split("\n") if line.strip())

        if not news_content.strip():
            logger.warning(f"Пустое содержимое статьи: {link}")
            return None

        # Извлекаем дату и время публикации со страницы
        fallback_datetime = kwargs.get("datetime")
        publication_datetime = self._extract_publication_datetime(soup, link, fallback_datetime)

        # Категория должна быть обязательно передана через kwargs
        category = kwargs.get("category")
        if not category:
            logger.warning(f"Категория не указана для новости {link}, используется 'Лента новостей'")
            category = "Лента новостей"
        
        return {
            "url": link,
            "title": title,
            "text": news_content.strip(),
            "publication_datetime": publication_datetime,
            "tags": None,
            "category": category,
        }

    async def parse(self) -> None:
        """
        Основной метод парсинга новостей.
        """
        logger.info("Начало парсинга RIA.ru")

        # Получаем даты периода парсинга из конфига
        start_date, end_date = self.get_parsing_dates()
        logger.info(f"Период парсинга: с {start_date} по {end_date}")

        # Получаем дату последней новости из БД
        last_news = self._get_last_news_datetime()
        if not last_news:
            # Устанавливаем очень старую дату в UTC, если нет предыдущих новостей
            last_news = datetime.fromtimestamp(0, tz=timezone.utc)
            logger.info("Предыдущих новостей не найдено, парсим все доступные")

        # Настройка SSL
        ssl_context = None if self.verify_ssl else False
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            # Определяем категории для парсинга
            categories_to_parse = self.categories if self.categories else [None]
            
            all_links = []
            
            # Парсим по каждой категории
            for category in categories_to_parse:
                category_name = category or "все"
                logger.info(f"Парсинг категории: {category_name}")
                
                # Получаем список ссылок для категории с учетом периода
                links = await self.fetch_links(session, category, start_date, end_date)
                
                if links:
                    all_links.extend(links)
                    logger.info(f"Получено {len(links)} новостей для категории '{category_name}'")
                else:
                    logger.warning(f"Не найдено ссылок для категории '{category_name}'")

            if not all_links:
                logger.warning("Не найдено ссылок на новости")
                return

            logger.info(f"Всего получено {len(all_links)} новостей из всех категорий")
            
            # Показываем диапазон дат полученных новостей
            if all_links:
                dates = [link["datetime"] for link in all_links]
                min_date = min(dates)
                max_date = max(dates)
                logger.info(f"Диапазон дат полученных новостей: с {min_date.date()} по {max_date.date()}")

            # Фильтруем по дате последней новости из БД
            links_before_last = len(all_links)
            if last_news:
                all_links = [link for link in all_links if link["datetime"] > last_news]
                logger.info(f"После фильтрации по последней новости из БД: {len(all_links)} из {links_before_last}")

            # Фильтруем по периоду из конфига (от начала первого дня до конца последнего дня)
            # Используем только дату для сравнения, чтобы включить все новости за день
            links_before_period = len(all_links)
            all_links = [
                link
                for link in all_links
                if start_date <= link["datetime"].date() <= end_date
            ]
            logger.info(f"После фильтрации по периоду ({start_date} - {end_date}): {len(all_links)} из {links_before_period}")
            
            links = all_links

            if not links:
                logger.info("Новых новостей не найдено")
                if links_before_period > 0:
                    logger.warning(
                        f"Найдено {links_before_period} новостей, но они не попадают в указанный период "
                        f"({start_date} - {end_date}). Возможно, нужно обновить даты в конфиге."
                    )
                return

            logger.info(f"Найдено {len(links)} новых новостей для парсинга")

            # Парсим новости параллельно
            tasks = [
                self.parse_news_item(
                    session,
                    link["link"],
                    **{k: v for k, v in link.items() if k != "link"}
                )
                for link in links
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Сохраняем новости в БД
            saved_count = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при парсинге новости: {result}")
                    continue

                if not result:
                    continue

                # Проверяем, не существует ли уже новость
                if self.db.news_exists(result["url"], result["category"]):
                    logger.debug(f"Новость уже существует: {result['url']}")
                    continue

                # Сохраняем в БД
                if self.db.save_news(
                    url=result["url"],
                    title=result["title"],
                    text=result["text"],
                    publication_datetime=result["publication_datetime"],
                    tags=result.get("tags"),
                    category=result["category"],
                ):
                    saved_count += 1
                    logger.debug(f"Сохранена новость в категорию '{result['category']}': {result['url']}")

            logger.info(f"Парсинг завершен. Сохранено новостей: {saved_count}")

    def close(self) -> None:
        """Закрытие подключения к БД."""
        self.db.close()


async def main():
    """Пример использования парсера."""
    parser = RIAParser()
    try:
        await parser.parse()
    finally:
        parser.close()


if __name__ == "__main__":
    asyncio.run(main())
