import logging
from typing import Optional

from core.kline_repo import latest_kline_1m
from core.pred_repo import upsert_pred_1m


def predict_next_minute_direction(symbol: str, lookback: int = 20) -> Optional[dict]:
    rows = latest_kline_1m(symbol, limit=lookback)
    if len(rows) < 5:
        return None

    closes = [float(r._mapping["close"]) for r in rows][::-1]  # 时间正序
    last_close = closes[-1]

    n = min(10, len(closes))
    sma = sum(closes[-n:]) / n
    score = last_close - sma
    direction = "UP" if score >= 0 else "DOWN"

    return {
        "symbol": symbol,
        "last_close": last_close,
        "sma": sma,
        "score": score,
        "direction": direction,
    }


def log_prediction(symbol: str, ts: str):
    """
    ts 必须传入：用刚写入K线的 ts（bar.ts_min），确保可回测对齐
    """
    pred = predict_next_minute_direction(symbol)
    if pred is None:
        logging.info("[预测] %s 数据不足，跳过", symbol)
        return

    upsert_pred_1m(
        symbol=symbol,
        ts=ts,
        horizon=1,
        direction=pred["direction"],
        score=float(pred["score"]),
        last_close=float(pred["last_close"]),
        sma=float(pred["sma"]),
        model="baseline_sma",
    )

    logging.info(
        "[预测] %s ts=%s next_1m=%s score=%.4f last_close=%.2f sma=%.2f ✅ 已写入 predictions_1m",
        pred["symbol"], ts, pred["direction"], pred["score"], pred["last_close"], pred["sma"]
    )
