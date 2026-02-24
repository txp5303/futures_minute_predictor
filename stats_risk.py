import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("data/market.db")

MODEL = "schemeA+"
SYMBOL = "螺纹"
HORIZON = 1


def compute_risk(rows):
    if not rows:
        print("无可用数据")
        return

    pnls = [r["pnl"] for r in rows]

    # ===== 最大连续亏损 =====
    max_streak = 0
    cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    # ===== 最大回撤 =====
    equity = []
    cum = 0
    for p in pnls:
        cum += p
        equity.append(cum)

    peak = float("-inf")
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    print("\n===== 风险结构分析（0.65 < p_up ≤ 0.90）=====")
    print(f"样本数: {len(pnls)}")
    print(f"最大连续亏损次数: {max_streak}")
    print(f"最大回撤: {max_dd:.4f}")
    print("=========================================")


def main():
    if not DB_PATH.exists():
        print("数据库不存在")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT ts, last_close, side
        FROM signals_1m
        WHERE model=? 
          AND symbol=? 
          AND p_up > 0.65 
          AND p_up <= 0.90
        ORDER BY ts ASC
        """,
        (MODEL, SYMBOL),
    ).fetchall()

    filled = []

    for ts, last_close, side in rows:
        dt0 = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        dt1 = dt0 + timedelta(minutes=HORIZON)
        ts1 = dt1.strftime("%Y-%m-%d %H:%M:%S")

        row = cur.execute(
            "SELECT close FROM kline_1m WHERE symbol=? AND ts=?",
            (SYMBOL, ts1),
        ).fetchone()

        if not row:
            continue

        future_close = float(row[0])
        last_close = float(last_close)

        pnl = (future_close - last_close) if side.upper() == "LONG" else (last_close - future_close)

        filled.append({"pnl": pnl})

    conn.close()

    compute_risk(filled)


if __name__ == "__main__":
    main()