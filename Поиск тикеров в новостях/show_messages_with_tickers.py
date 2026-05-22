#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт для просмотра сообщений с указанными тикерами"""

import sqlite3
import json
from pathlib import Path

# Пути к файлам
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "telegram_messages.db"


def find_messages_with_tickers(conn, tickers_to_find):
    """Находит все сообщения с указанными тикерами"""
    cursor = conn.cursor()
    
    # Находим все таблицы с новостями
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name != 'message_embeddings' "
        "AND name != 'duplicates'"
    )
    tables = [row[0] for row in cursor.fetchall()]
    
    all_messages = []
    
    for table_name in tables:
        # Проверяем, есть ли колонка tickers
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'tickers' not in columns:
            continue
        
        # Ищем сообщения с нужными тикерами
        cursor.execute(
            f"SELECT message_id, message, tickers FROM {table_name} "
            f"WHERE tickers IS NOT NULL AND tickers != ''"
        )
        rows = cursor.fetchall()
        
        for message_id, message, tickers_json in rows:
            try:
                tickers = json.loads(tickers_json)
                # Проверяем, есть ли хотя бы один из искомых тикеров
                found_tickers = [t for t in tickers if t in tickers_to_find]
                if found_tickers:
                    all_messages.append({
                        'table': table_name,
                        'message_id': message_id,
                        'message': message,
                        'tickers': tickers,
                        'found_tickers': found_tickers
                    })
            except (json.JSONDecodeError, TypeError):
                continue
    
    return all_messages


def main():
    """Основная функция"""
    print("=" * 80)
    print("ПОИСК СООБЩЕНИЙ С ТИКЕРАМИ X5 И T")
    print("=" * 80)
    
    # Подключаемся к БД
    if not DB_PATH.exists():
        print(f"База данных не найдена: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    # Ищем сообщения с тикерами X5 и T
    tickers_to_find = ["X5", "T"]
    messages = find_messages_with_tickers(conn, tickers_to_find)
    
    conn.close()
    
    # Группируем по тикерам
    messages_by_ticker = {"X5": [], "T": [], "BOTH": []}
    
    for msg in messages:
        found = set(msg['found_tickers'])
        if "X5" in found and "T" in found:
            messages_by_ticker["BOTH"].append(msg)
        elif "X5" in found:
            messages_by_ticker["X5"].append(msg)
        elif "T" in found:
            messages_by_ticker["T"].append(msg)
    
    # Выводим результаты
    print(f"\nНайдено сообщений:")
    print(f"   С тикером X5: {len(messages_by_ticker['X5'])}")
    print(f"   С тикером T: {len(messages_by_ticker['T'])}")
    print(f"   С обоими тикерами (X5 и T): {len(messages_by_ticker['BOTH'])}")
    print(f"   Всего уникальных сообщений: {len(messages)}")
    
    # Выводим сообщения с тикером X5
    if messages_by_ticker["X5"]:
        print(f"\n{'=' * 80}")
        print(f"СООБЩЕНИЯ С ТИКЕРОМ X5 ({len(messages_by_ticker['X5'])} шт.)")
        print(f"{'=' * 80}")
        for i, msg in enumerate(messages_by_ticker["X5"], 1):
            print(f"\n[{i}] Таблица: {msg['table']} | ID: {msg['message_id']}")
            print(f"    Тикеры: {', '.join(msg['tickers'])}")
            print(f"    Сообщение:")
            print(f"    {msg['message'][:500]}{'...' if len(msg['message']) > 500 else ''}")
            print("-" * 80)
    
    # Выводим сообщения с тикером T
    if messages_by_ticker["T"]:
        print(f"\n{'=' * 80}")
        print(f"СООБЩЕНИЯ С ТИКЕРОМ T ({len(messages_by_ticker['T'])} шт.)")
        print(f"{'=' * 80}")
        for i, msg in enumerate(messages_by_ticker["T"], 1):
            print(f"\n[{i}] Таблица: {msg['table']} | ID: {msg['message_id']}")
            print(f"    Тикеры: {', '.join(msg['tickers'])}")
            print(f"    Сообщение:")
            print(f"    {msg['message'][:500]}{'...' if len(msg['message']) > 500 else ''}")
            print("-" * 80)
    
    # Выводим сообщения с обоими тикерами
    if messages_by_ticker["BOTH"]:
        print(f"\n{'=' * 80}")
        print(f"СООБЩЕНИЯ С ОБОИМИ ТИКЕРАМИ (X5 и T) ({len(messages_by_ticker['BOTH'])} шт.)")
        print(f"{'=' * 80}")
        for i, msg in enumerate(messages_by_ticker["BOTH"], 1):
            print(f"\n[{i}] Таблица: {msg['table']} | ID: {msg['message_id']}")
            print(f"    Тикеры: {', '.join(msg['tickers'])}")
            print(f"    Сообщение:")
            print(f"    {msg['message'][:500]}{'...' if len(msg['message']) > 500 else ''}")
            print("-" * 80)
    
    if not messages:
        print("\nСообщения с указанными тикерами не найдены.")


if __name__ == "__main__":
    main()
