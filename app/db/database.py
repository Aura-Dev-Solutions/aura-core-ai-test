import sqlite3

DB_PATH = "app/documents.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                text TEXT,
                embedding TEXT,
                category TEXT,
                entities TEXT
            )
        ''')
        conn.commit()
