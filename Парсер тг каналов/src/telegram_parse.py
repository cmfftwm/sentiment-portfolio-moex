import configparser
import datetime
import os
import sqlite3
import re
from pathlib import Path
from tqdm.asyncio import tqdm

from telethon import TelegramClient
from telethon.tl.types import MessageEntityHashtag
from datetime import timezone, timedelta

# Определяем корневую директорию проекта (на уровень выше src)
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Создаем папку для метаданных сессии
metadata_dir = project_root / 'metadata'
metadata_dir.mkdir(exist_ok=True)

# Грузим конфиг
config = configparser.ConfigParser()

config_path = project_root / 'configs' / 'tg.ini'
if not config_path.exists():
    raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

config.read(str(config_path), encoding='utf-8')

# Проверяем, что секция DATA существует
if 'DATA' not in config:
    raise KeyError(f"Секция 'DATA' не найдена в файле {config_path}")

# Безопасный парсинг списка каналов
channels_str = config['DATA']['channels'].strip()
if channels_str.startswith('[') and channels_str.endswith(']'):
    # Убираем квадратные скобки и разбиваем по запятым
    channels_str = channels_str[1:-1]
    channels = [ch.strip().strip("'\"") for ch in channels_str.split(',')]
    # Убираем пустые строки
    channels = [ch for ch in channels if ch]
else:
    channels = [channels_str] if channels_str else []

end_date_str = config['DATA'].get('end_date', 'None')
start_date_str = config['DATA'].get('start_date', 'None')
save_root = config['DATA']['save_root']
reverse = config['DATA'].getboolean('reverse', fallback=False)

# Конфиг с приватными ключами
secret_config_path = project_root / 'configs' / 'tg_secret.ini'
if not secret_config_path.exists():
    print(f"ВНИМАНИЕ: Файл {secret_config_path} не найден!")
    print("Создайте файл configs/tg_secret.ini со следующим содержимым:")
    print("[DATA_SECRET]")
    print("api_id = ваш_api_id")
    print("api_hash = ваш_api_hash")
    print("phone = ваш_номер_телефона")
    print("username = ваш_username")
    raise FileNotFoundError(f"Файл {secret_config_path} не найден. Создайте его с вашими учетными данными Telegram API.")

config.read(str(secret_config_path), encoding='utf-8')

# Проверяем, что секция DATA_SECRET существует
if 'DATA_SECRET' not in config:
    raise KeyError(f"Секция 'DATA_SECRET' не найдена в файле {secret_config_path}")

api_id = config['DATA_SECRET']['api_id']
api_hash = config['DATA_SECRET']['api_hash']
phone = config['DATA_SECRET']['phone']
username = config['DATA_SECRET'].get('username', '')

# Обработка конечной даты
if end_date_str == 'None' or not end_date_str:
    end_date = datetime.datetime.now(timezone.utc).replace(second=0, hour=0, minute=0)
elif reverse:
    end_date = datetime.datetime.strptime(end_date_str, "%d-%m-%Y").replace(tzinfo=timezone.utc) + timedelta(days=1)
else: 
    end_date = datetime.datetime.strptime(end_date_str, "%d-%m-%Y").replace(tzinfo=timezone.utc)

# Обработка начальной даты
if start_date_str == 'None' or not start_date_str:
    start_date = None
else:
    start_date = datetime.datetime.strptime(start_date_str, "%d-%m-%Y").replace(tzinfo=timezone.utc)

# Путь к файлу сессии
session_path = metadata_dir / 'tg_parse'
client = TelegramClient(str(session_path), api_id, api_hash, system_version='4.16.30-vxCUSTOM')

def extract_hashtags(message):
    """Извлекает хештеги из сообщения Telegram"""
    hashtags = []
    
    # Проверяем наличие entities (более надежный способ)
    if hasattr(message, 'entities') and message.entities:
        for entity in message.entities:
            if isinstance(entity, MessageEntityHashtag):
                # Извлекаем хештег из текста по offset и length
                start = entity.offset
                end = entity.offset + entity.length
                if message.message and len(message.message) >= end:
                    hashtag = message.message[start:end]
                    hashtags.append(hashtag)
    
    # Если entities нет, парсим текст регулярным выражением
    if not hashtags and message.message:
        hashtags_found = re.findall(r'#\w+', message.message)
        hashtags.extend(hashtags_found)
    
    # Убираем дубликаты и возвращаем строку через запятую
    unique_hashtags = list(set(hashtags))
    return ', '.join(unique_hashtags) if unique_hashtags else None

def sanitize_table_name(channel_name: str) -> str:
    """Очищает имя канала для использования в имени таблицы SQLite"""
    # Заменяем недопустимые символы на подчеркивания
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', channel_name)
    # Убираем множественные подчеркивания
    table_name = re.sub(r'_+', '_', table_name)
    # Убираем подчеркивания в начале и конце
    table_name = table_name.strip('_')
    # Если имя пустое, используем дефолтное
    if not table_name:
        table_name = 'channel'
    return f'messages_{table_name}'

def create_channel_table(cursor, table_name: str):
    """Создает таблицу для конкретного канала"""
    # Создаем таблицу для сообщений канала
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            message_number INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            date TIMESTAMP NOT NULL,
            message TEXT,
            hashtags TEXT
        )
    ''')
    
    # Создаем индексы для быстрого поиска
    cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_{table_name}_date 
        ON {table_name}(date)
    ''')
    
    cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_{table_name}_number 
        ON {table_name}(message_number)
    ''')
    
    cursor.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_{table_name}_message_id 
        ON {table_name}(message_id)
    ''')

async def main(phone, channels, save_root, reverse=reverse):
    await client.start(phone=phone)
    print("Client Created")

    # Создаем папку для сохранения, если её нет
    save_path = Path(save_root)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Создаем или подключаемся к базе данных
    db_filename = 'telegram_messages.db'
    db_path = save_path / db_filename
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print(f"База данных: {db_path}")
    print(f"Всего каналов для парсинга: {len(channels)}")
    print("-" * 50)

    for channel in channels:
        if not channel or not channel.strip():
            continue
        try:
            my_channel = await client.get_entity(channel)
            channel_name = channel.split('/')[-1] if '/' in channel else channel
            print(f'\nПарсим канал: {channel_name}')
            
            # Создаем таблицу для этого канала
            table_name = sanitize_table_name(channel_name)
            create_channel_table(cursor, table_name)
            conn.commit()
            print(f'Таблица: {table_name}')

            messages_count = 0
            message_number = 0
            
            async for message in tqdm(client.iter_messages(entity=my_channel,
                                                    offset_date=end_date - timedelta(days=1), 
                                                    reverse=reverse),
                                    desc=f"Парсинг {channel_name}"):
                # Убеждаемся, что дата в UTC
                if message.date.tzinfo is None:
                    message_date = message.date.replace(tzinfo=timezone.utc)
                else:
                    message_date = message.date.astimezone(timezone.utc)
                
                # Проверяем условие по дате
                if reverse:
                    # От новых к старым
                    if end_date and message_date >= end_date:
                        break  # Превысили верхнюю границу
                    if start_date and message_date < start_date:
                        break  # Вышли за нижнюю границу диапазона
                else: 
                    # От старых к новым
                    if end_date and message_date > end_date:
                        break  # Превысили верхнюю границу
                    if start_date and message_date < start_date:
                        continue  # Пропускаем сообщения до start_date, еще не дошли до диапазона
                
                # Увеличиваем номер сообщения
                message_number += 1
                
                # Получаем ID сообщения из Telegram
                telegram_message_id = message.id if hasattr(message, 'id') else None
                
                # Получаем текст сообщения
                message_text = message.message if hasattr(message, 'message') else None
                
                # Извлекаем хештеги
                hashtags = extract_hashtags(message)
                
                # Сохраняем в базу данных в таблицу канала
                cursor.execute(f'''
                    INSERT INTO {table_name} (message_number, message_id, date, message, hashtags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (message_number, telegram_message_id, message_date.isoformat(), message_text, hashtags))
                
                messages_count += 1
                
                # Коммитим каждые 100 сообщений для оптимизации
                if messages_count % 100 == 0:
                    conn.commit()
            
            # Финальный коммит
            conn.commit()
            
            if messages_count > 0:
                print(f'Сохранено {messages_count} сообщений из канала {channel_name} в базу данных')
            else:
                print(f'Не найдено сообщений для канала {channel_name} в указанном диапазоне дат')
                
        except Exception as e:
            print(f'Ошибка при парсинге канала {channel}: {e}')
            continue
    
    # Закрываем соединение с базой данных
    conn.close()
    print(f'\nПарсинг завершен. Данные сохранены в {db_path}')

if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main(phone, channels, save_root, reverse))