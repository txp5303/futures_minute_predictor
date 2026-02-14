import sqlite3

DB_PATH = "data/market.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) 列出所有表
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    print("DB:", DB_PATH)
    print("Tables:", tables)

    # 2) 每个表的行数
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t};")
            n = cur.fetchone()[0]
            print(f"{t}: {n}")
        except Exception as e:
            print(f"{t}: count failed -> {e}")

    conn.close()

if __name__ == "__main__":
    main()
