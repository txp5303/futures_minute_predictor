import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "market.db"

sql = """
CREATE TABLE IF NOT EXISTS decisions_1m (
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,

    close REAL,
    sma REAL,

    direction TEXT,
    p_up REAL,
    score REAL,

    side TEXT,
    strength TEXT,

    stage2_ok INTEGER,
    stage2_reason TEXT,
    vol_mean_abs REAL,
    jump_abs REAL,

    risk_paused INTEGER,
    risk_reason TEXT,

    confirm_n INTEGER,
    confirm_k INTEGER,
    confirm_count INTEGER,
    confirm_ok INTEGER,
    confirm_reason TEXT,

    emitted INTEGER,
    emit_side TEXT,
    emit_strength TEXT,
    cooldown_skipped INTEGER,

    created_at TEXT NOT NULL,

    PRIMARY KEY(symbol, ts, model)
);
"""

conn = sqlite3.connect(str(DB_PATH))
conn.execute(sql)
conn.commit()
conn.close()
print("OK: decisions_1m created in", DB_PATH)