"""
Асинхронный парсер новостей с сайта Коммерсантъ.
Парсит архив рубрик по дням: Политика, Экономика, Финансы, Потребрынок.
"""
import asyncio
import configparser
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from dateutil.parser import parse

from db_handler import NewsDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

RUBRIC_IDS = {
    "Политика": 2,
    "Экономика": 3,
    "Финансы": 40,
    "Потребрынок": 41,
}


class CommersantParser:
    """Асинхронный парсер новостей Коммерсантъ."""

    def __init__(self, config_path: str = "config.ini"):
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_path, encoding="utf-8")

        categories_str = self.config.get("commersant", "categories", fallback="")
        self.categories = [c.strip() for c in categories_str.split(",") if c.strip()]

        self.start_date = datetime.fromisoformat(
            self.config.get("commersant", "start_date", fallback="2020-01-01")
        ).replace(tzinfo=timezone.utc)
        self.end_date = datetime.fromisoformat(
            self.config.get("commersant", "end_date", fallback="2025-11-20")
        ).replace(tzinfo=timezone.utc)

        self.timeout = self.config.getint("parsers", "request_timeout", fallback=30)
        self.retries = self.config.getint("parsers", "retries", fallback=3)
        self.delay = self.config.getfloat("parsers", "request_delay", fallback=2.0)

        db_path = self.config.get("database", "commersant_db_path", fallback="commersant_news.db")
        self.db = NewsDatabase(db_path)

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        try:
            # Формат на странице: "15.01.2020, 23:58"
            dt = parse(date_str, dayfirst=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    async def _make_request(self, url: str) -> Optional[str]:
        for attempt in range(self.retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        headers=HEADERS,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ssl=False,
                    ) as response:
                        if response.status == 200:
                            return await response.text()
                        logger.warning(f"Статус {response.status} для {url}")
            except Exception as e:
                logger.error(f"Ошибка запроса (попытка {attempt + 1}): {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    def _date_range(self):
        """Генератор дат от start_date до end_date включительно."""
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)

    async def fetch_links_for_day(self, category: str, rubric_id: int, date: datetime) -> List[Dict]:
        url = f"https://www.kommersant.ru/archive/rubric/{rubric_id}/day/{date.strftime('%Y-%m-%d')}"
        response = await self._make_request(url)
        if not response:
            return []

        soup = BeautifulSoup(response, "html.parser")
        news_data = []

        for article in soup.find_all("article", attrs={"data-article-url": True}):
            link = article.get("data-article-url", "").strip()
            title = article.get("data-article-title", "").strip()
            if not link or not title:
                continue

            date_tag = article.find("p", class_="uho__tag")
            if not date_tag:
                continue
            pub_date = self._parse_date(date_tag.text.strip())
            if not pub_date:
                continue

            news_data.append(
                {
                    "link": link,
                    "title": title,
                    "description": "",
                    "category": category,
                    "datetime": pub_date,
                }
            )

        return news_data

    async def parse_news_item(self, link: str, **kwargs) -> bool:
        if self.db.news_exists(link, kwargs["category"]):
            logger.debug(f"Уже существует: {link}")
            return False

        response = await self._make_request(link)
        if not response:
            logger.error(f"Ошибка при запросе страницы: {link}")
            return False

        soup = BeautifulSoup(response, "html.parser")
        article = soup.find("article")
        if not article:
            logger.warning(f"Не найден тег <article> на странице: {link}")
            return False

        news_content = "\n".join(p.text for p in article.find_all("p"))

        saved = self.db.save_news(
            url=link,
            title=kwargs["title"],
            text=news_content,
            publication_datetime=kwargs["datetime"],
            tags=kwargs.get("description") or None,
            category=kwargs["category"],
        )

        if saved:
            logger.info(f"Сохранена: {kwargs['title'][:60]}")
        return saved

    async def parse_category(self, category: str) -> int:
        rubric_id = RUBRIC_IDS.get(category)
        if not rubric_id:
            logger.warning(f"Неизвестная рубрика: {category}")
            return 0

        logger.info(f"=== Начинаем рубрику: {category} ===")
        saved_count = 0

        for date in self._date_range():
            links = await self.fetch_links_for_day(category, rubric_id, date)

            for item in links:
                saved = await self.parse_news_item(**item)
                if saved:
                    saved_count += 1
                await asyncio.sleep(self.delay)

            if links:
                logger.info(
                    f"[{category}] {date.strftime('%Y-%m-%d')}: "
                    f"{len(links)} новостей, сохранено: {saved_count}"
                )

        logger.info(f"[{category}] Итого сохранено: {saved_count}")
        return saved_count

    async def parse(self) -> int:
        results = await asyncio.gather(
            *[self.parse_category(cat) for cat in self.categories]
        )
        total_saved = sum(results)
        logger.info(f"Всего сохранено: {total_saved}")
        return total_saved


if __name__ == "__main__":
    print("started")
    parser = CommersantParser()
    asyncio.run(parser.parse())
