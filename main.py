# main.py
# -*- coding: utf-8 -*-
"""
futures_minute_predictor - main entry

Scheme A+（强化版）信号策略：
- 只保留：L3 / LONG
- 其余全部禁用（包含所有 SHORT、以及 L1/L2 的 LONG）
- 保持原阈值分层不变 + cooldown
  （即：L3 LONG >= 0.57 才会出信号；A+只吃这一档）

数据库：data/market.db
兼容点：
- predictions_1m / signals_1m：自动探测 created_at / model 列是否存在
- 若表缺少 (symbol, ts) 的 PRIMARY KEY/UNIQUE，避免 ON CONFLICT 报错：
  改为 “先 UPDATE 再 INSERT” 的兼容写法
"""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

from apscheduler.schedulers.blocking import BlockingScheduler


# =========================
# Config
# =========================
DB_PATH = Path("data/market.db")
SYMBOL = "螺纹"

COLLECT_INTERVAL_SEC = 2
PREDICT_INTERVAL_SEC = 10

SIGNAL_COOLDOWN_SEC = 30

MODEL_NAME = "schemeA_plus_simple_v1"

# 阈值分层：从强到弱（L3->L2->L1）
PUP_THRESHOLDS = {
    "L1": {"LONG": 0.53, "SHORT": 0.47},
    "L2": {"LONG": 0.55, "SHORT": 0.45},
    "L3": {"LONG": 0.57, "SHORT": 0.43},
}

# 方案A+：只保留这一档
KEEP_ONLY = ("L3", "LONG")


# =========================
# Logging
# =========================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# =========================
# DB helpers
# =========================
def ensure_db() -> None:
    """只保证 kline_1m 存在；predictions_1m / signals_1m 由你原库提供（此处不强行改结构）。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kline_1m (
                symbol TEXT NOT NULL,
                ts TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY(symbol, ts)
            )
            """
        )
        conn.commit()


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cur = conn.cursor()
    cols = set()
    try:
        rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    except sqlite3.OperationalError:
        return set()
    for r in rows:
        cols.add(r[1])
    return cols


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row)


def has_unique_on_cols(conn: sqlite3.Connection, table: str, cols: tuple[str, ...]) -> bool:
    """
    判断表是否存在覆盖 cols 的 PRIMARY KEY 或 UNIQUE 索引。
    用于决定能否安全使用 ON CONFLICT(cols...).
    """
    cur = conn.cursor()

    # 1) primary key?
    try:
        info = cur.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return False

    pk_cols: List[str] = []
    for r in info:
        # r: cid, name, type, notnull, dflt_value, pk (pk is 0/1/2...)
        if r[5]:
            pk_cols.append(r[1])
    if tuple(pk_cols) == cols:
        return True

    # 2) unique indexes?
    try:
        idx_list = cur.execute(f"PRAGMA index_list({table})").fetchall()
    except sqlite3.OperationalError:
        return False

    # idx_list row: seq, name, unique, origin, partial
    for (_seq, idx_name, unique, _origin, _partial) in idx_list:
        if not unique:
            continue
        try:
            idx_info = cur.execute(f"PRAGMA index_info({idx_name})").fetchall()
        except sqlite3.OperationalError:
            continue
        # idx_info row: seqno, cid, name
        idx_cols = tuple([r[2] for r in idx_info])
        if idx_cols == cols:
            return True

    return False


def upsert_kline(symbol: str, ts: str, o: float, h: float, l: float, c: float, v: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO kline_1m(symbol, ts, open, high, low, close, volume)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(symbol, ts) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            (symbol, ts, o, h, l, c, v),
        )
        conn.commit()


def get_latest_kline(symbol: str) -> Optional[Tuple[str, str, float, float]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT symbol, ts, close, volume FROM kline_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return row


def load_recent_kline(symbol: str, limit: int = 300) -> list[Tuple[str, float]]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT ts, close FROM kline_1m WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        rows.reverse()
        return [(r[0], float(r[1])) for r in rows]


def safe_update_then_insert(
    conn: sqlite3.Connection,
    table: str,
    key_cols: tuple[str, str],  # (symbol_col, ts_col)
    key_vals: tuple[str, str],
    set_cols: list[str],
    set_vals: list,
    insert_cols: list[str],
    insert_vals: list,
) -> None:
    """
    在没有 UNIQUE/PK 的情况下，避免 ON CONFLICT 报错：
    先 UPDATE（可能更新多行，但在你的逻辑里 symbol+ts 应该唯一）
    若 UPDATE rowcount==0 再 INSERT。
    """
    cur = conn.cursor()

    set_clause = ", ".join([f"{c}=?" for c in set_cols])
    where_clause = f"{key_cols[0]}=? AND {key_cols[1]}=?"
    cur.execute(
        f"UPDATE {table} SET {set_clause} WHERE {where_clause}",
        (*set_vals, *key_vals),
    )
    if cur.rowcount == 0:
        cols_clause = ", ".join(insert_cols)
        qmarks = ", ".join(["?"] * len(insert_cols))
        cur.execute(
            f"INSERT INTO {table} ({cols_clause}) VALUES ({qmarks})",
            insert_vals,
        )


def upsert_prediction(symbol: str, ts: str, direction: str, score: float, p_up: float, last_close: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, "predictions_1m"):
            return

        cols = table_columns(conn, "predictions_1m")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        use_created_at = "created_at" in cols
        use_model = "model" in cols

        base_cols = ["symbol", "ts", "direction", "score", "p_up", "last_close"]
        base_vals = [symbol, ts, direction, score, p_up, last_close]

        extra_cols = []
        extra_vals = []
        if use_model:
            extra_cols.append("model")
            extra_vals.append(MODEL_NAME)
        if use_created_at:
            extra_cols.append("created_at")
            extra_vals.append(now_str)

        insert_cols = base_cols + extra_cols
        insert_vals = base_vals + extra_vals

        # UPDATE 时：不更新 key；其余都更新
        set_cols = ["direction", "score", "p_up", "last_close"] + extra_cols
        set_vals = [direction, score, p_up, last_close] + extra_vals

        # 如果存在唯一约束，就用 ON CONFLICT；否则降级为 update->insert
        if has_unique_on_cols(conn, "predictions_1m", ("symbol", "ts")):
            cur = conn.cursor()
            cols_clause = ", ".join(insert_cols)
            qmarks = ", ".join(["?"] * len(insert_cols))
            set_clause = ", ".join([f"{c}=excluded.{c}" for c in set_cols])
            cur.execute(
                f"""
                INSERT INTO predictions_1m({cols_clause})
                VALUES({qmarks})
                ON CONFLICT(symbol, ts) DO UPDATE SET
                    {set_clause}
                """,
                insert_vals,
            )
        else:
            safe_update_then_insert(
                conn=conn,
                table="predictions_1m",
                key_cols=("symbol", "ts"),
                key_vals=(symbol, ts),
                set_cols=set_cols,
                set_vals=set_vals,
                insert_cols=insert_cols,
                insert_vals=insert_vals,
            )

        conn.commit()


def upsert_signal(symbol: str, ts: str, side: str, strength: str, p_up: float, score: float, last_close: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, "signals_1m"):
            return

        cols = table_columns(conn, "signals_1m")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        use_created_at = "created_at" in cols
        use_model = "model" in cols  # 你的库里 model 可能 NOT NULL

        base_cols = ["symbol", "ts", "side", "strength", "p_up", "score", "last_close"]
        base_vals = [symbol, ts, side, strength, p_up, score, last_close]

        extra_cols = []
        extra_vals = []
        if use_model:
            extra_cols.append("model")
            extra_vals.append(MODEL_NAME)
        if use_created_at:
            extra_cols.append("created_at")
            extra_vals.append(now_str)

        insert_cols = base_cols + extra_cols
        insert_vals = base_vals + extra_vals

        set_cols = ["side", "strength", "p_up", "score", "last_close"] + extra_cols
        set_vals = [side, strength, p_up, score, last_close] + extra_vals

        if has_unique_on_cols(conn, "signals_1m", ("symbol", "ts")):
            cur = conn.cursor()
            cols_clause = ", ".join(insert_cols)
            qmarks = ", ".join(["?"] * len(insert_cols))
            set_clause = ", ".join([f"{c}=excluded.{c}" for c in set_cols])
            cur.execute(
                f"""
                INSERT INTO signals_1m({cols_clause})
                VALUES({qmarks})
                ON CONFLICT(symbol, ts) DO UPDATE SET
                    {set_clause}
                """,
                insert_vals,
            )
        else:
            safe_update_then_insert(
                conn=conn,
                table="signals_1m",
                key_cols=("symbol", "ts"),
                key_vals=(symbol, ts),
                set_cols=set_cols,
                set_vals=set_vals,
                insert_cols=insert_cols,
                insert_vals=insert_vals,
            )

        conn.commit()


def get_last_signal_ts(symbol: str) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, "signals_1m"):
            return None
        cur = conn.cursor()
        row = cur.execute(
            "SELECT ts FROM signals_1m WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return row[0] if row else None


# =========================
# Demo Data Source (replace with your real source)
# =========================
class DemoMarketDataSource:
    """每2秒生成一个tick，聚合成1分钟K线（demo用）。"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._last_price = 3030.0 + random.random() * 30.0
        self._bar_ts: Optional[datetime] = None
        self._o = self._h = self._l = self._c = self._last_price
        self._v = 0.0

    @staticmethod
    def _floor_to_minute(dt: datetime) -> datetime:
        return dt.replace(second=0, microsecond=0)

    def collect_and_maybe_emit_bar(self) -> Optional[Tuple[str, str, float, float, float, float, float]]:
        now = datetime.now()
        price = self._last_price + random.uniform(-2.5, 2.5)
        vol = random.randint(1, 80)

        bar_ts = self._floor_to_minute(now)
        if self._bar_ts is None:
            self._bar_ts = bar_ts
            self._o = self._h = self._l = self._c = price
            self._v = vol
        elif bar_ts != self._bar_ts:
            emitted = (
                self.symbol,
                self._bar_ts.strftime("%Y-%m-%d %H:%M:%S"),
                float(self._o),
                float(self._h),
                float(self._l),
                float(self._c),
                float(self._v),
            )
            self._bar_ts = bar_ts
            self._o = self._h = self._l = self._c = price
            self._v = vol
            self._last_price = price
            return emitted
        else:
            self._c = price
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._v += vol

        self._last_price = price
        return None


# =========================
# Simple Predictor (replace with your real ML)
# =========================
class SimplePredictor:
    def predict(self, closes: list[float]) -> Optional[Tuple[str, float, float]]:
        if len(closes) < 30:
            return None
        m1 = closes[-1] - closes[-5]
        m2 = closes[-1] - closes[-15]
        raw = 0.7 * m1 + 0.3 * m2 + random.uniform(-1.5, 1.5)
        p_up = 1.0 / (1.0 + pow(2.718281828, -raw / 8.0))
        score = (p_up - 0.5) * 2.0
        direction = "UP" if p_up >= 0.5 else "DOWN"
        return direction, float(p_up), float(score)


# =========================
# Signal logic (Scheme A+)
# =========================
@dataclass
class Signal:
    side: str       # LONG / SHORT
    strength: str   # L1 / L2 / L3


def determine_strength(p_up: float) -> Optional[Signal]:
    # 依然按分层算出强度，但 A+ 只会保留 L3/LONG
    if p_up >= PUP_THRESHOLDS["L3"]["LONG"]:
        return Signal("LONG", "L3")
    if p_up >= PUP_THRESHOLDS["L2"]["LONG"]:
        return Signal("LONG", "L2")
    if p_up >= PUP_THRESHOLDS["L1"]["LONG"]:
        return Signal("LONG", "L1")

    if p_up <= PUP_THRESHOLDS["L3"]["SHORT"]:
        return Signal("SHORT", "L3")
    if p_up <= PUP_THRESHOLDS["L2"]["SHORT"]:
        return Signal("SHORT", "L2")
    if p_up <= PUP_THRESHOLDS["L1"]["SHORT"]:
        return Signal("SHORT", "L1")

    return None


def scheme_a_plus_filter(sig: Signal) -> Optional[Signal]:
    # A+：只保留 L3/LONG
    if (sig.strength, sig.side) == KEEP_ONLY:
        return sig
    return None


def cooldown_ok(symbol: str, bar_dt: datetime) -> bool:
    last_ts_str = get_last_signal_ts(symbol)
    if not last_ts_str:
        return True
    try:
        last_dt = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return True
    return (bar_dt - last_dt).total_seconds() >= SIGNAL_COOLDOWN_SEC


def log_thresholds() -> None:
    base = (
        f"Signal thresholds: "
        f"L1(LONG>={PUP_THRESHOLDS['L1']['LONG']:.2f}, SHORT<={PUP_THRESHOLDS['L1']['SHORT']:.2f}), "
        f"L2(LONG>={PUP_THRESHOLDS['L2']['LONG']:.2f}, SHORT<={PUP_THRESHOLDS['L2']['SHORT']:.2f}), "
        f"L3(LONG>={PUP_THRESHOLDS['L3']['LONG']:.2f}, SHORT<={PUP_THRESHOLDS['L3']['SHORT']:.2f}), "
        f"cooldown={SIGNAL_COOLDOWN_SEC}s, model={MODEL_NAME}"
    )
    logging.info("%s", base)
    logging.info("Scheme A+ enabled: KEEP ONLY = %s/%s", KEEP_ONLY[1], KEEP_ONLY[0])


# =========================
# Scheduler Jobs
# =========================
ds = DemoMarketDataSource(SYMBOL)
predictor = SimplePredictor()
_last_pred_bar_ts: Optional[str] = None


def job_collect() -> None:
    bar = ds.collect_and_maybe_emit_bar()
    if not bar:
        return
    symbol, ts, o, h, l, c, v = bar
    upsert_kline(symbol, ts, o, h, l, c, v)


def job_predict() -> None:
    global _last_pred_bar_ts

    row = get_latest_kline(SYMBOL)
    if not row:
        logging.info("[PRED] no kline yet.")
        return

    symbol, bar_ts, last_close, _volume = row
    if _last_pred_bar_ts == bar_ts:
        return
    _last_pred_bar_ts = bar_ts

    rows = load_recent_kline(symbol, limit=300)
    closes = [c for (_ts, c) in rows]
    res = predictor.predict(closes)
    if not res:
        logging.info("[PRED] features not ready yet. rows=%d symbol=%s", len(rows), symbol)
        return

    direction, p_up, score = res

    # 写预测（兼容 created_at/model/unique）
    upsert_prediction(symbol, bar_ts, direction, score, p_up, float(last_close))
    logging.info("[PRED] %s %s dir=%s p_up=%.4f score=%.6f", bar_ts, symbol, direction, p_up, score)

    # 生成信号（A+ 过滤：只保留 L3/LONG）
    sig0 = determine_strength(p_up)
    if not sig0:
        logging.info("[SIGNAL] %s %s NONE (p_up=%.4f)", bar_ts, symbol, p_up)
        return

    sig = scheme_a_plus_filter(sig0)
    if not sig:
        logging.info(
            "[SIGNAL] %s %s SUPPRESSED by SchemeA+ (%s/%s) (p_up=%.4f score=%.6f) keep_only=%s/%s",
            bar_ts, symbol, sig0.side, sig0.strength, p_up, score, KEEP_ONLY[1], KEEP_ONLY[0]
        )
        return

    bar_dt = datetime.strptime(bar_ts, "%Y-%m-%d %H:%M:%S")
    if not cooldown_ok(symbol, bar_dt):
        logging.info("[SIGNAL] %s %s COOLING DOWN -> NONE", bar_ts, symbol)
        return

    upsert_signal(symbol, bar_ts, sig.side, sig.strength, p_up, score, float(last_close))
    logging.info("[SIGNAL] %s %s %s/%s p_up=%.4f score=%.6f", bar_ts, symbol, sig.side, sig.strength, p_up, score)


def main() -> None:
    setup_logging()
    ensure_db()

    sched = BlockingScheduler(timezone="Asia/Shanghai")
    sched.add_job(job_collect, "interval", seconds=COLLECT_INTERVAL_SEC, id="job_collect", max_instances=1)
    sched.add_job(job_predict, "interval", seconds=PREDICT_INTERVAL_SEC, id="job_predict", max_instances=1)

    logging.info("Scheduler started")
    logging.info("System running. Ctrl+C to stop.")
    log_thresholds()

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler has been shut down")
        try:
            sched.shutdown(wait=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()