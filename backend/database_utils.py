import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # User table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        # History table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                image_name TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Database initialized at {DB_PATH}")
    except Exception as e:
        print(f"Database initialization error: {e}")

def save_prediction(username, image_name, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (username, image_name, prediction, confidence) VALUES (?, ?, ?, ?)",
        (username, image_name, prediction, confidence)
    )
    conn.commit()
    conn.close()

def get_history(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, prediction, confidence, timestamp FROM predictions WHERE username = ? ORDER BY timestamp DESC", (username,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def create_user(username, password_hash):
    try:
        if not username or not password_hash:
            raise ValueError("Username and password cannot be empty")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        conn.close()
        print(f"User {username} created successfully")
        return True
    except sqlite3.IntegrityError as e:
        print(f"Integrity error creating user: {e}")
        return False
    except Exception as e:
        print(f"Error creating user: {e}")
        raise

def get_user_hash(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

if __name__ == "__main__":
    init_db()
