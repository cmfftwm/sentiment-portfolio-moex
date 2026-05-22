import sqlite3
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Пути к файлам
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "telegram_messages.db"

# Колонки для удаления
COLUMNS_TO_REMOVE = ["hashtags", "tickers"]

# Создавать ли резервную копию перед удалением
CREATE_BACKUP = True


def get_table_columns(cursor, table_name):
    """Получает список колонок таблицы"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def check_columns_exist(cursor, table_name, columns):
    """Проверяет, существуют ли указанные колонки в таблице"""
    existing_columns = get_table_columns(cursor, table_name)
    return [col for col in columns if col in existing_columns]


def drop_columns_sqlite_modern(cursor, table_name, columns_to_remove):
    """Удаляет колонки используя ALTER TABLE DROP COLUMN (SQLite 3.35.0+)"""
    for col in columns_to_remove:
        try:
            cursor.execute(f"ALTER TABLE {table_name} DROP COLUMN {col}")
            print(f"      Удалена колонка '{col}'")
        except sqlite3.OperationalError as e:
            if "no such column" in str(e).lower():
                print(f"      Колонка '{col}' не существует")
            else:
                raise


def recreate_table_without_columns(conn, cursor, table_name, columns_to_remove):
    """Пересоздает таблицу без указанных колонок (для старых версий SQLite)"""
    print(f"      Пересоздаю таблицу без колонок {columns_to_remove}...")
    
    # Получаем все колонки
    all_columns = get_table_columns(cursor, table_name)
    columns_to_keep = [col for col in all_columns if col not in columns_to_remove]
    
    if len(columns_to_keep) == len(all_columns):
        print(f"      Колонки {columns_to_remove} не найдены в таблице")
        return False
    
    # Получаем данные (только нужные колонки)
    columns_str = ", ".join(columns_to_keep)
    cursor.execute(f"SELECT {columns_str} FROM {table_name}")
    rows = cursor.fetchall()
    row_count = len(rows)
    
    # Получаем информацию о колонках для создания структуры
    cursor.execute(f"PRAGMA table_info({table_name})")
    column_info = cursor.fetchall()
    
    # Создаем временную таблицу
    temp_table = f"{table_name}_temp"
    
    # Формируем CREATE TABLE statement
    create_parts = []
    primary_keys = []
    
    for col_info in column_info:
        col_name = col_info[1]
        if col_name in columns_to_remove:
            continue
        
        col_type = col_info[2]
        not_null = "NOT NULL" if col_info[3] else ""
        default_val = ""
        if col_info[4] is not None:
            if isinstance(col_info[4], str):
                default_val = f"DEFAULT '{col_info[4]}'"
            else:
                default_val = f"DEFAULT {col_info[4]}"
        
        is_pk = col_info[5]
        
        col_def_parts = [col_name, col_type]
        if not_null:
            col_def_parts.append(not_null)
        if default_val:
            col_def_parts.append(default_val)
        
        create_parts.append(" ".join(col_def_parts))
        
        if is_pk:
            primary_keys.append(col_name)
    
    # Добавляем PRIMARY KEY если есть
    if primary_keys:
        create_parts.append(f"PRIMARY KEY ({', '.join(primary_keys)})")
    
    create_sql_new = f"CREATE TABLE {temp_table} ({', '.join(create_parts)})"
    
    # Создаем временную таблицу
    cursor.execute(create_sql_new)
    
    # Копируем данные
    if rows:
        placeholders = ", ".join(["?" for _ in columns_to_keep])
        insert_sql = f"INSERT INTO {temp_table} ({columns_str}) VALUES ({placeholders})"
        cursor.executemany(insert_sql, rows)
    
    # Удаляем старую таблицу
    cursor.execute(f"DROP TABLE {table_name}")
    
    # Переименовываем временную таблицу
    cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
    
    print(f"      Таблица пересоздана, скопировано строк: {row_count}, удалено колонок: {len(columns_to_remove)}")
    return True


def remove_columns_from_table(conn, cursor, table_name, columns_to_remove):
    """Удаляет колонки из таблицы"""
    # Проверяем, какие колонки существуют
    existing_columns = check_columns_exist(cursor, table_name, columns_to_remove)
    
    if not existing_columns:
        print(f"      Колонки {columns_to_remove} не найдены в таблице")
        return 0
    
    # Проверяем версию SQLite
    cursor.execute("SELECT sqlite_version()")
    sqlite_version = cursor.fetchone()[0]
    version_parts = [int(x) for x in sqlite_version.split('.')]
    supports_drop_column = (version_parts[0] > 3 or 
                           (version_parts[0] == 3 and version_parts[1] >= 35))
    
    if supports_drop_column:
        # Используем современный способ
        print(f"      SQLite {sqlite_version} поддерживает DROP COLUMN")
        drop_columns_sqlite_modern(cursor, table_name, existing_columns)
        return len(existing_columns)
    else:
        # Используем пересоздание таблицы
        print(f"      SQLite {sqlite_version} не поддерживает DROP COLUMN, пересоздаю таблицу")
        if recreate_table_without_columns(conn, cursor, table_name, existing_columns):
            return len(existing_columns)
        return 0


def create_backup(db_path):
    """Создает резервную копию базы данных"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}.db"
    
    print(f"\nСоздание резервной копии...")
    try:
        shutil.copy2(db_path, backup_path)
        print(f"Резервная копия создана: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Не удалось создать резервную копию: {e}")
        return None


def main():
    """Основная функция"""
    print("=" * 60)
    print("УДАЛЕНИЕ КОЛОНОК ИЗ БАЗЫ ДАННЫХ")
    print("=" * 60)
    print(f"Колонки для удаления: {', '.join(COLUMNS_TO_REMOVE)}")
    
    # Проверяем существование БД
    if not DB_PATH.exists():
        print(f"\nБаза данных не найдена: {DB_PATH}")
        return
    
    # Создаем резервную копию
    if CREATE_BACKUP:
        backup_path = create_backup(DB_PATH)
        if not backup_path:
            response = input("\nПродолжить без резервной копии? (yes/no): ")
            if response.lower() not in ['yes', 'y', 'да', 'д']:
                print("Операция отменена")
                return
    
    print(f"\nПодключение к базе данных {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Проверяем версию SQLite
    cursor.execute("SELECT sqlite_version()")
    sqlite_version = cursor.fetchone()[0]
    print(f"Версия SQLite: {sqlite_version}")
    
    # Находим все таблицы с новостями (исключаем служебные)
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name != 'message_embeddings' "
        "AND name != 'duplicates'"
    )
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\nНайдено таблиц: {len(tables)}")
    print(f"  Таблицы: {', '.join(tables)}")
    
    # Обрабатываем каждую таблицу
    total_removed = 0
    for table in tables:
        print(f"\n{'=' * 60}")
        print(f"Обработка таблицы: {table}")
        print(f"{'=' * 60}")
        
        removed = remove_columns_from_table(conn, cursor, table, COLUMNS_TO_REMOVE)
        total_removed += removed
        
        conn.commit()
    
    conn.close()
    
    print(f"\n{'=' * 60}")
    print(f"ГОТОВО!")
    print(f"   Всего удалено колонок: {total_removed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
