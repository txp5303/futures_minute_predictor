from sqlalchemy import text
from core.db import ENGINE

def upsert_kline_1m(
    symbol: str,
    ts: str,
    o: float, h: float, l: float, c: float,
    volume: float,
    open_interest: float | None = None,
    source: str | None = "demo",
):
    """
    写入1分钟K线（同symbol+ts 自动去重：已存在则忽略）
    """
    sql = """
    INSERT OR IGNORE INTO kline_1m
    (symbol, ts, open, high, low, close, volume, open_interest, source)
    VALUES
    (:symbol, :ts, :open, :high, :low, :close, :volume, :open_interest, :source)
    """
    with ENGINE.begin() as conn:
        conn.execute(
            text(sql),
            dict(
                symbol=symbol, ts=ts,
                open=o, high=h, low=l, close=c,
                volume=volume,
                open_interest=open_interest,
                source=source,
            ),
        )

def latest_kline_1m(symbol: str, limit: int = 5):
    """
    读取最近N条分钟K线
    """
    sql = """
    SELECT symbol, ts, open, high, low, close, volume, open_interest, source, created_at
    FROM kline_1m
    WHERE symbol = :symbol
    ORDER BY ts DESC
    LIMIT :limit
    """
    with ENGINE.begin() as conn:
        rows = conn.execute(text(sql), dict(symbol=symbol, limit=limit)).fetchall()
    return rows
