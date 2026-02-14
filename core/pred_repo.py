from sqlalchemy import text
from core.db import ENGINE


def upsert_pred_1m(
    symbol: str,
    ts: str,
    horizon: int,
    direction: str,
    score: float,
    last_close: float,
    sma: float,
    model: str = "baseline_sma",
):
    sql = """
    INSERT INTO predictions_1m (symbol, ts, horizon, direction, score, last_close, sma, model)
    VALUES (:symbol, :ts, :horizon, :direction, :score, :last_close, :sma, :model)
    ON CONFLICT(symbol, ts, horizon, model) DO UPDATE SET
        direction=excluded.direction,
        score=excluded.score,
        last_close=excluded.last_close,
        sma=excluded.sma
    """
    with ENGINE.begin() as conn:
        conn.execute(text(sql), dict(
            symbol=symbol, ts=ts, horizon=horizon,
            direction=direction, score=score,
            last_close=last_close, sma=sma, model=model
        ))
