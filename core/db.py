from pathlib import Path
from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "market.db"


def init_db():
    """
    重要：SQLite 建表语句必须作为“完整 statement”执行
    不能按行拆开执行，否则会出现 near "open": syntax error
    """
    engine = create_engine(f"sqlite:///{DB_PATH}")

    schema_sql = """
    CREATE TABLE IF NOT EXISTS kline_1m (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        open_interest REAL,
        source TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
        UNIQUE(symbol, ts)
    );

    CREATE INDEX IF NOT EXISTS idx_kline_1m_symbol_ts ON kline_1m(symbol, ts);

    CREATE TABLE IF NOT EXISTS predictions_1m (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        direction TEXT NOT NULL,
        score REAL NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
        UNIQUE(ts, horizon, model)
    );

    CREATE INDEX IF NOT EXISTS idx_predictions_1m_ts ON predictions_1m(ts);

    CREATE TABLE IF NOT EXISTS tick (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        price REAL,
        volume REAL,
        open_interest REAL,
        source TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
    );

    CREATE INDEX IF NOT EXISTS idx_tick_symbol_ts ON tick(symbol, ts);
    """

    # 用原生 driver 执行整段 SQL（多语句）
    with engine.begin() as conn:
        for stmt in [s.strip() for s in schema_sql.split(";")]:
            if stmt:
                conn.exec_driver_sql(stmt + ";")


if __name__ == "__main__":
    init_db()
    print(f"DB initialized: {DB_PATH}")
