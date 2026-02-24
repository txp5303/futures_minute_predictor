# -*- coding: utf-8 -*-
"""
scheduler_app.py (FULL)
中国期货分钟级实时预测系统（个人使用版）
主链路：采集tick -> 聚合1mK线 -> 预测p_up -> Stage2过滤与风控 -> Stage3确认 -> signals_1m + decisions_1m落库
并包含：decisions回填(pnl/hit) + 滚动监控(PF/Exp/连亏)

说明：
- 本文件尽量“单文件可跑”，不依赖 confirm.py 等外部模块，避免你之前的导入/粘贴损坏问题。
- 默认 Demo 数据源（随机游走）保证你随时能验证链路；如你有真实行情源，可替换 DataSource.get_tick()。

运行：
  python scheduler_app.py --symbol 螺纹 --use_demo 1
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from apscheduler.schedulers.background import BackgroundScheduler


# =========================
# Paths / Logging
# =========================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "data" / "market.db")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def ensure_dir(path: str):
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Config (Stage2 / Stage3 / Model)
# =========================

# --- Scheduler frequency ---
COLLECT_INTERVAL_SEC = 2
PREDICT_INTERVAL_SEC = 10
BACKFILL_INTERVAL_SEC = 30
MONITOR_INTERVAL_SEC = 60

# --- Trading symbol ---
DEFAULT_SYMBOL = "螺纹"

# --- Prediction horizon (minutes) ---
HORIZON_MIN = 1

# --- Model name (写入DB用于隔离统计) ---
MODEL_NAME = "schemeA+_with_decisions_v1"

# --- Stage2 filter ---
# ONLY_SIDE: "LONG" / "SHORT" / "BOTH"
ONLY_SIDE = "BOTH"

# strength threshold
MIN_STRENGTH = "L2"  # allow L2/L3; "L1" means allow all

# p_up band for signals (neutral zone outside bands is dropped)
# 例：p_up>=0.65 -> LONG; p_up<=0.35 -> SHORT; 中间不发
PUP_LONG_TH = 0.65
PUP_SHORT_TH = 0.35

# 风控：触发暂停（秒）
RISK_PAUSE_SEC = 60

# 风控：连续亏损阈值（基于 decisions 回填）
MAX_CONSEC_LOSS = 4

# 风控：回撤监控窗口（用于滚动监控输出，不直接拦截）
MONITOR_WINDOW = 20

# --- Stage3 confirm ---
# 需要连续 N 次候选信号同方向才确认
CONFIRM_N = 2
# 确认窗口有效期（分钟）：超过窗口则重新warming
CONFIRM_WINDOW_MIN = 3


# =========================
# Utils
# =========================

def now_ts_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def strength_rank(s: str) -> int:
    s = (s or "").upper()
    if s == "L3":
        return 3
    if s == "L2":
        return 2
    return 1  # L1 or unknown


def rank_to_strength(r: int) -> str:
    if r >= 3:
        return "L3"
    if r == 2:
        return "L2"
    return "L1"


def sigmoid(x: float) -> float:
    # safe sigmoid
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# =========================
# DB Layer
# =========================

def get_conn() -> sqlite3.Connection:
    ensure_dir(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    # ticks
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tick (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        price REAL NOT NULL,
        volume INTEGER,
        open_interest INTEGER,
        source TEXT,
        PRIMARY KEY(symbol, ts)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tick_ts ON tick(ts);")

    # 1m kline
    cur.execute("""
    CREATE TABLE IF NOT EXISTS kline_1m (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER,
        open_interest INTEGER,
        source TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY(symbol, ts)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kline_ts ON kline_1m(ts);")

    # predictions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions_1m (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        direction TEXT NOT NULL,
        score REAL,
        p_up REAL,
        model TEXT NOT NULL,
        last_close REAL,
        sma REAL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY(symbol, ts, horizon, model)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions_1m(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions_1m(model);")

    # signals
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals_1m (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        side TEXT NOT NULL,              -- LONG/SHORT
        strength TEXT NOT NULL,          -- L1/L2/L3
        p_up REAL,
        score REAL,
        model TEXT NOT NULL,
        future_close REAL,
        pnl REAL,
        hit_flag INTEGER,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY(symbol, ts, horizon, model)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sig_ts ON signals_1m(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sig_model ON signals_1m(model);")

    # decisions (Decision Layer)
    _ensure_decisions_table(conn)

    conn.commit()


# =========================
# Decision Layer (decisions_1m)
# =========================

def _ensure_decisions_table(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS decisions_1m (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        horizon INTEGER NOT NULL DEFAULT 1,

        side TEXT NOT NULL,              -- LONG / SHORT
        strength TEXT NOT NULL,          -- L1 / L2 / L3
        p_up REAL,
        score REAL,
        model TEXT NOT NULL DEFAULT 'unknown',

        stage2_pass INTEGER NOT NULL DEFAULT 1,
        stage3_pass INTEGER NOT NULL DEFAULT 1,

        reason TEXT,
        risk_pause_until TEXT,

        status TEXT NOT NULL DEFAULT 'OPEN',  -- OPEN / CLOSED / CANCELLED
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),

        future_close REAL,
        pnl REAL,
        hit_flag INTEGER,
        closed_at TEXT,

        PRIMARY KEY (symbol, ts, horizon, model)
    );
    """)

    # 补列兼容（避免你之前手动建表字段不一致）
    cur.execute("PRAGMA table_info(decisions_1m);")
    cols = {r[1] for r in cur.fetchall()}

    def add_col(col_name: str, ddl: str):
        if col_name not in cols:
            cur.execute(f"ALTER TABLE decisions_1m ADD COLUMN {ddl};")

    add_col("horizon", "horizon INTEGER NOT NULL DEFAULT 1")
    add_col("side", "side TEXT NOT NULL DEFAULT 'LONG'")
    add_col("strength", "strength TEXT NOT NULL DEFAULT 'L1'")
    add_col("p_up", "p_up REAL")
    add_col("score", "score REAL")
    add_col("model", "model TEXT NOT NULL DEFAULT 'unknown'")
    add_col("stage2_pass", "stage2_pass INTEGER NOT NULL DEFAULT 1")
    add_col("stage3_pass", "stage3_pass INTEGER NOT NULL DEFAULT 1")
    add_col("reason", "reason TEXT")
    add_col("risk_pause_until", "risk_pause_until TEXT")
    add_col("status", "status TEXT NOT NULL DEFAULT 'OPEN'")
    add_col("created_at", "created_at TEXT NOT NULL DEFAULT (datetime('now'))")
    add_col("updated_at", "updated_at TEXT NOT NULL DEFAULT (datetime('now'))")
    add_col("future_close", "future_close REAL")
    add_col("pnl", "pnl REAL")
    add_col("hit_flag", "hit_flag INTEGER")
    add_col("closed_at", "closed_at TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions_1m(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_model ON decisions_1m(model);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions_1m(status);")

    conn.commit()


def upsert_decision(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    ts: str,
    horizon: int,
    side: str,
    strength: str,
    p_up: Optional[float],
    score: Optional[float],
    model: str,
    stage2_pass: int = 1,
    stage3_pass: int = 1,
    reason: Optional[str] = None,
    risk_pause_until: Optional[str] = None,
    status: str = "OPEN",
):
    _ensure_decisions_table(conn)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO decisions_1m
    (symbol, ts, horizon, side, strength, p_up, score, model,
     stage2_pass, stage3_pass, reason, risk_pause_until, status,
     created_at, updated_at)
    VALUES
    (?, ?, ?, ?, ?, ?, ?, ?,
     ?, ?, ?, ?, ?,
     datetime('now'), datetime('now'))
    ON CONFLICT(symbol, ts, horizon, model) DO UPDATE SET
      side=excluded.side,
      strength=excluded.strength,
      p_up=excluded.p_up,
      score=excluded.score,
      stage2_pass=excluded.stage2_pass,
      stage3_pass=excluded.stage3_pass,
      reason=excluded.reason,
      risk_pause_until=excluded.risk_pause_until,
      status=excluded.status,
      updated_at=datetime('now')
    ;
    """, (
        symbol, ts, int(horizon), side, strength, p_up, score, model,
        int(stage2_pass), int(stage3_pass), reason, risk_pause_until, status
    ))
    conn.commit()


def backfill_decisions_outcome(
    conn: sqlite3.Connection,
    *,
    lookback_limit: int = 2000,
) -> int:
    _ensure_decisions_table(conn)
    cur = conn.cursor()

    cur.execute("""
    SELECT symbol, ts, horizon, side, model
    FROM decisions_1m
    WHERE status='OPEN' AND (future_close IS NULL)
    ORDER BY ts ASC
    LIMIT ?;
    """, (int(lookback_limit),))
    rows = cur.fetchall()
    if not rows:
        return 0

    filled = 0
    for symbol, ts, horizon, side, model in rows:
        try:
            base_dt = datetime.fromisoformat(ts)
        except Exception:
            continue

        future_dt = base_dt + timedelta(minutes=int(horizon))
        future_ts = future_dt.strftime("%Y-%m-%d %H:%M:%S")

        cur.execute("SELECT close FROM kline_1m WHERE symbol=? AND ts=?;", (symbol, ts))
        r0 = cur.fetchone()
        if not r0:
            continue
        close0 = float(r0[0])

        cur.execute("SELECT close FROM kline_1m WHERE symbol=? AND ts=?;", (symbol, future_ts))
        r1 = cur.fetchone()
        if not r1:
            continue
        close1 = float(r1[0])

        side_u = str(side).upper()
        if side_u == "SHORT":
            pnl = close0 - close1
            hit = 1 if close1 < close0 else 0
        else:
            pnl = close1 - close0
            hit = 1 if close1 > close0 else 0

        cur.execute("""
        UPDATE decisions_1m
        SET future_close=?,
            pnl=?,
            hit_flag=?,
            status='CLOSED',
            closed_at=datetime('now'),
            updated_at=datetime('now')
        WHERE symbol=? AND ts=? AND horizon=? AND model=?;
        """, (close1, pnl, int(hit), symbol, ts, int(horizon), model))

        filled += 1

    conn.commit()
    return filled


def rolling_decisions_monitor(
    conn: sqlite3.Connection,
    *,
    model: str,
    window: int = 20,
) -> Dict[str, Any]:
    _ensure_decisions_table(conn)
    cur = conn.cursor()

    cur.execute("""
    SELECT pnl
    FROM decisions_1m
    WHERE model=? AND status='CLOSED' AND pnl IS NOT NULL
    ORDER BY ts DESC
    LIMIT ?;
    """, (model, int(window)))
    pnls = [float(x[0]) for x in cur.fetchall()]
    if not pnls:
        return {"model": model, "window": window, "count": 0}

    s = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    if losses and sum(losses) > 1e-12:
        pf = sum(wins) / sum(losses) if wins else 0.0
    else:
        pf = float("inf") if wins else 0.0

    exp = s / len(pnls)

    consec_loss = 0
    for p in pnls:
        if p < 0:
            consec_loss += 1
        else:
            break

    return {
        "model": model,
        "window": window,
        "count": len(pnls),
        "sum_pnl": s,
        "PF": pf,
        "Exp": exp,
        "consec_loss": consec_loss,
    }


def get_consec_loss(conn: sqlite3.Connection, model: str) -> int:
    """最近开始向后数连续亏损次数（基于 decisions CLOSED）"""
    cur = conn.cursor()
    cur.execute("""
    SELECT pnl
    FROM decisions_1m
    WHERE model=? AND status='CLOSED' AND pnl IS NOT NULL
    ORDER BY ts DESC
    LIMIT 200;
    """, (model,))
    rows = cur.fetchall()
    consec = 0
    for (pnl,) in rows:
        p = float(pnl)
        if p < 0:
            consec += 1
        else:
            break
    return consec


# =========================
# Market Data Source (Demo)
# =========================

@dataclass
class Tick:
    symbol: str
    ts: str
    price: float
    volume: int = 0
    open_interest: int = 0
    source: str = "DEMO"


class DemoMarketDataSource:
    def __init__(self, symbol: str, start_price: float = 3000.0):
        self.symbol = symbol
        self.price = float(start_price)
        self.vol = 0
        self.oi = 0

    def get_tick(self) -> Tick:
        # random walk
        step = random.uniform(-2.0, 2.0)
        self.price = max(1.0, self.price + step)
        self.vol += random.randint(1, 50)
        self.oi += random.randint(-3, 3)

        ts = now_ts_str()
        return Tick(symbol=self.symbol, ts=ts, price=float(self.price), volume=int(self.vol), open_interest=int(self.oi))


# =========================
# Minute Aggregator (tick -> 1m OHLC)
# =========================

@dataclass
class Bar1m:
    symbol: str
    ts: str  # minute ts (YYYY-mm-dd HH:MM:SS) with seconds=00
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int
    source: str


class MinuteAggregator:
    def __init__(self):
        self._cur_minute: Optional[str] = None
        self._o = self._h = self._l = self._c = None
        self._last_total_vol: Optional[int] = None
        self._vol_delta: int = 0
        self._oi: int = 0
        self._symbol: str = ""
        self._source: str = ""

    def update(self, tick: Tick) -> Tuple[Optional[Bar1m], Optional[Bar1m]]:
        """
        返回 (bar_snap, bar_flushed)
        bar_snap：当前分钟快照（不一定落库也行，但我们这里会落库覆盖）
        bar_flushed：当分钟切换时，上一分钟flush出来的bar
        """
        dt = datetime.fromisoformat(tick.ts)
        minute_dt = floor_to_minute(dt)
        minute_ts = minute_dt.strftime("%Y-%m-%d %H:%M:%S")

        bar_flushed = None

        if self._cur_minute is None:
            # init
            self._cur_minute = minute_ts
            self._symbol = tick.symbol
            self._source = tick.source
            self._o = self._h = self._l = self._c = tick.price
            self._last_total_vol = tick.volume
            self._vol_delta = 0
            self._oi = tick.open_interest
        elif minute_ts != self._cur_minute:
            # flush old minute
            bar_flushed = Bar1m(
                symbol=self._symbol,
                ts=self._cur_minute,
                open=float(self._o),
                high=float(self._h),
                low=float(self._l),
                close=float(self._c),
                volume=int(self._vol_delta),
                open_interest=int(self._oi),
                source=self._source,
            )
            # reset for new minute
            self._cur_minute = minute_ts
            self._symbol = tick.symbol
            self._source = tick.source
            self._o = self._h = self._l = self._c = tick.price
            # volume delta: reset base
            self._last_total_vol = tick.volume
            self._vol_delta = 0
            self._oi = tick.open_interest
        else:
            # same minute update
            self._h = max(self._h, tick.price)
            self._l = min(self._l, tick.price)
            self._c = tick.price
            self._oi = tick.open_interest
            # volume delta
            if self._last_total_vol is None:
                self._last_total_vol = tick.volume
            else:
                d = tick.volume - self._last_total_vol
                if d >= 0:
                    self._vol_delta += d
                self._last_total_vol = tick.volume

        bar_snap = Bar1m(
            symbol=self._symbol,
            ts=self._cur_minute,
            open=float(self._o),
            high=float(self._h),
            low=float(self._l),
            close=float(self._c),
            volume=int(self._vol_delta),
            open_interest=int(self._oi),
            source=self._source,
        )

        return bar_snap, bar_flushed


def upsert_tick(conn: sqlite3.Connection, t: Tick):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO tick(symbol, ts, price, volume, open_interest, source)
    VALUES(?, ?, ?, ?, ?, ?)
    ON CONFLICT(symbol, ts) DO UPDATE SET
      price=excluded.price,
      volume=excluded.volume,
      open_interest=excluded.open_interest,
      source=excluded.source
    ;
    """, (t.symbol, t.ts, float(t.price), int(t.volume), int(t.open_interest), t.source))
    conn.commit()


def upsert_kline_1m(conn: sqlite3.Connection, b: Bar1m):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO kline_1m(symbol, ts, open, high, low, close, volume, open_interest, source, created_at, updated_at)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ON CONFLICT(symbol, ts) DO UPDATE SET
      open=excluded.open,
      high=excluded.high,
      low=excluded.low,
      close=excluded.close,
      volume=excluded.volume,
      open_interest=excluded.open_interest,
      source=excluded.source,
      updated_at=datetime('now')
    ;
    """, (
        b.symbol, b.ts, float(b.open), float(b.high), float(b.low), float(b.close),
        int(b.volume), int(b.open_interest), b.source
    ))
    conn.commit()


def get_latest_bar(conn: sqlite3.Connection, symbol: str) -> Optional[Tuple[str, float]]:
    cur = conn.cursor()
    cur.execute("""
    SELECT ts, close
    FROM kline_1m
    WHERE symbol=?
    ORDER BY ts DESC
    LIMIT 1;
    """, (symbol,))
    r = cur.fetchone()
    if not r:
        return None
    return str(r[0]), float(r[1])


def load_recent_kline(conn: sqlite3.Connection, symbol: str, limit: int = 300) -> List[Tuple[str, float]]:
    cur = conn.cursor()
    cur.execute("""
    SELECT ts, close
    FROM kline_1m
    WHERE symbol=?
    ORDER BY ts DESC
    LIMIT ?;
    """, (symbol, int(limit)))
    rows = cur.fetchall()
    rows.reverse()
    return [(str(ts), float(close)) for ts, close in rows]


def upsert_prediction(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    ts: str,
    horizon: int,
    direction: str,
    score: float,
    p_up: float,
    model: str,
    last_close: float,
    sma: float,
):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO predictions_1m(symbol, ts, horizon, direction, score, p_up, model, last_close, sma, created_at, updated_at)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ON CONFLICT(symbol, ts, horizon, model) DO UPDATE SET
      direction=excluded.direction,
      score=excluded.score,
      p_up=excluded.p_up,
      last_close=excluded.last_close,
      sma=excluded.sma,
      updated_at=datetime('now')
    ;
    """, (symbol, ts, int(horizon), direction, float(score), float(p_up), model, float(last_close), float(sma)))
    conn.commit()


def upsert_signal(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    ts: str,
    horizon: int,
    side: str,
    strength: str,
    p_up: float,
    score: float,
    model: str,
):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO signals_1m(symbol, ts, horizon, side, strength, p_up, score, model, created_at, updated_at)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    ON CONFLICT(symbol, ts, horizon, model) DO UPDATE SET
      side=excluded.side,
      strength=excluded.strength,
      p_up=excluded.p_up,
      score=excluded.score,
      updated_at=datetime('now')
    ;
    """, (symbol, ts, int(horizon), side, strength, float(p_up), float(score), model))
    conn.commit()


# =========================
# Predictor (simple, stable)
# =========================

class SimplePredictor:
    """
    用收敛的连续概率：p_up = sigmoid(k * (close - sma) / scale)
    score = p_up - 0.5（或直接用差值也行）
    """
    def __init__(self, sma_n: int = 20, k: float = 3.0, scale: float = 10.0):
        self.sma_n = int(sma_n)
        self.k = float(k)
        self.scale = float(scale)

    def predict(self, closes: List[float]) -> Optional[Tuple[str, float, float, float]]:
        """
        returns: (direction, p_up, score, sma)
        """
        if len(closes) < self.sma_n:
            return None
        sma = sum(closes[-self.sma_n:]) / self.sma_n
        last = closes[-1]
        x = self.k * ((last - sma) / max(1e-6, self.scale))
        p_up = sigmoid(x)
        score = p_up - 0.5
        direction = "UP" if p_up >= 0.5 else "DOWN"
        return direction, float(p_up), float(score), float(sma)


# =========================
# Stage2 Filter + Risk Gate
# =========================

@dataclass
class CandidateSignal:
    symbol: str
    ts: str
    horizon: int
    side: str         # LONG / SHORT
    strength: str     # L1/L2/L3
    p_up: float
    score: float
    model: str


class RiskState:
    def __init__(self):
        self.paused_until: Optional[datetime] = None

    def is_paused(self) -> bool:
        if self.paused_until is None:
            return False
        return datetime.now() < self.paused_until

    def pause(self, seconds: int):
        self.paused_until = datetime.now() + timedelta(seconds=int(seconds))

    def paused_until_str(self) -> Optional[str]:
        if self.paused_until is None:
            return None
        return self.paused_until.strftime("%Y-%m-%d %H:%M:%S")


def stage2_filter_and_risk_gate(
    conn: sqlite3.Connection,
    risk: RiskState,
    cand: CandidateSignal,
) -> Tuple[bool, str]:
    """
    returns: (pass?, reason)
    """

    # risk pause
    if risk.is_paused():
        return False, f"risk_pause: paused_until={risk.paused_until_str()}"

    # ONLY_SIDE
    if ONLY_SIDE.upper() in ("LONG", "SHORT"):
        if cand.side.upper() != ONLY_SIDE.upper():
            return False, f"only_side={ONLY_SIDE}"

    # strength
    if strength_rank(cand.strength) < strength_rank(MIN_STRENGTH):
        return False, f"filter_strength<{MIN_STRENGTH}"

    # p_up neutral band
    if cand.side.upper() == "LONG":
        if cand.p_up < PUP_LONG_TH:
            return False, "no_signal_band_long"
    elif cand.side.upper() == "SHORT":
        if cand.p_up > PUP_SHORT_TH:
            return False, "no_signal_band_short"
    else:
        return False, "bad_side"

    # 风控：连续亏损过多 -> pause
    consec = get_consec_loss(conn, cand.model)
    if consec >= MAX_CONSEC_LOSS:
        risk.pause(RISK_PAUSE_SEC)
        return False, f"risk_pause_triggered: consec_loss={consec} pause={RISK_PAUSE_SEC}s"

    return True, "ok"


def strength_from_pup(p_up: float) -> str:
    """
    简单分级：越极端越强。
    你可以按你已有的L1/L2/L3规则替换。
    """
    if p_up >= 0.80 or p_up <= 0.20:
        return "L3"
    if p_up >= 0.70 or p_up <= 0.30:
        return "L2"
    return "L1"


def side_from_pup(p_up: float) -> Optional[str]:
    if p_up >= PUP_LONG_TH:
        return "LONG"
    if p_up <= PUP_SHORT_TH:
        return "SHORT"
    return None


# =========================
# Stage3 Confirm
# =========================

class ConfirmEngine:
    """
    连续N次候选信号同 side 才确认。
    同时：窗口超过 CONFIRM_WINDOW_MIN 分钟会 reset。
    """
    def __init__(self, n: int, window_min: int):
        self.n = int(n)
        self.window_min = int(window_min)
        self._buf: List[CandidateSignal] = []
        self._window_start: Optional[datetime] = None

    def _reset(self):
        self._buf.clear()
        self._window_start = None

    def update(self, cand: CandidateSignal) -> Tuple[bool, str]:
        try:
            dt = datetime.fromisoformat(cand.ts)
        except Exception:
            return False, "bad_ts"

        if self._window_start is None:
            self._window_start = dt
            self._buf = [cand]
            return False, f"warming_up 1/{self.n}"

        if dt - self._window_start > timedelta(minutes=self.window_min):
            # window expired
            self._reset()
            self._window_start = dt
            self._buf = [cand]
            return False, f"window_expired_reset 1/{self.n}"

        # require same side
        if self._buf and self._buf[-1].side.upper() != cand.side.upper():
            # direction changed -> reset buffer within same window
            self._buf = [cand]
            return False, f"side_changed_reset 1/{self.n}"

        self._buf.append(cand)
        if len(self._buf) >= self.n:
            # confirmed
            self._reset()
            return True, f"confirmed N={self.n}"
        return False, f"warming_up {len(self._buf)}/{self.n}"


# =========================
# App State
# =========================

class AppState:
    def __init__(self, symbol: str, use_demo: bool = True):
        self.symbol = symbol
        self.conn = get_conn()
        init_db(self.conn)

        self.ds = DemoMarketDataSource(symbol=symbol) if use_demo else DemoMarketDataSource(symbol=symbol)
        self.agg = MinuteAggregator()
        self.pred = SimplePredictor(sma_n=20, k=4.0, scale=12.0)

        self.risk = RiskState()
        self.confirm = ConfirmEngine(n=CONFIRM_N, window_min=CONFIRM_WINDOW_MIN)

        self._last_pred_bar_ts: Optional[str] = None

        logging.info("[INIT] DB_PATH=%s", DB_PATH)
        logging.info("[INIT] symbol=%s model=%s horizon=%s", self.symbol, MODEL_NAME, HORIZON_MIN)
        logging.info("[INIT] Stage2 ONLY_SIDE=%s MIN_STRENGTH=%s PUP_LONG_TH=%.2f PUP_SHORT_TH=%.2f",
                     ONLY_SIDE, MIN_STRENGTH, PUP_LONG_TH, PUP_SHORT_TH)
        logging.info("[INIT] Stage3 CONFIRM_N=%d WINDOW_MIN=%d", CONFIRM_N, CONFIRM_WINDOW_MIN)


# =========================
# Jobs
# =========================

def job_collect_and_aggregate(st: AppState):
    t = st.ds.get_tick()
    upsert_tick(st.conn, t)

    bar_snap, bar_flushed = st.agg.update(t)

    # always upsert snapshot (cover same minute)
    if bar_snap:
        upsert_kline_1m(st.conn, bar_snap)
        logging.info("[BAR_SNAP] ts=%s close=%.2f vol=%d", bar_snap.ts, bar_snap.close, bar_snap.volume)

    if bar_flushed:
        # flushed bar already upserted by snapshot, but we log it for clarity
        logging.info("[COLLECT] flushed_ts=%s close=%.2f", bar_flushed.ts, bar_flushed.close)


def job_predict_and_decide(st: AppState):
    # heartbeat
    latest = get_latest_bar(st.conn, st.symbol)
    if not latest:
        logging.info("[ML] no bar yet")
        return

    bar_ts, last_close = latest

    # 你若希望“同一分钟只做一次”，保留这一段；
    # 若想分钟内滚动预测，把这段注释掉即可。
    if st._last_pred_bar_ts == bar_ts:
        logging.info("[ML] same minute snapshot (skip) ts=%s", bar_ts)
        return
    st._last_pred_bar_ts = bar_ts

    rows = load_recent_kline(st.conn, st.symbol, limit=300)
    closes = [c for _, c in rows]
    pred = st.pred.predict(closes)
    if not pred:
        logging.info("[ML] features not ready. rows=%d", len(rows))
        return

    direction, p_up, score, sma = pred
    upsert_prediction(
        st.conn,
        symbol=st.symbol,
        ts=bar_ts,
        horizon=HORIZON_MIN,
        direction=direction,
        score=score,
        p_up=p_up,
        model=MODEL_NAME,
        last_close=last_close,
        sma=sma,
    )
    logging.info("[ML] ts=%s last_close=%.2f sma=%.2f dir=%s p_up=%.4f score=%.4f",
                 bar_ts, last_close, sma, direction, p_up, score)

    # candidate
    side = side_from_pup(p_up)
    if side is None:
        logging.info("[STAGE2] no_signal_band p_up=%.4f", p_up)
        return

    strength = strength_from_pup(p_up)
    cand = CandidateSignal(
        symbol=st.symbol,
        ts=bar_ts,
        horizon=HORIZON_MIN,
        side=side,
        strength=strength,
        p_up=p_up,
        score=score,
        model=MODEL_NAME,
    )

    # Stage2
    ok2, reason2 = stage2_filter_and_risk_gate(st.conn, st.risk, cand)
    if not ok2:
        logging.info("[STAGE2] skip(%s) ts=%s side=%s strength=%s p_up=%.4f",
                     reason2, bar_ts, side, strength, p_up)
        return

    logging.info("[STAGE2] pass ts=%s side=%s strength=%s p_up=%.4f", bar_ts, side, strength, p_up)

    # Stage3 confirm
    ok3, reason3 = st.confirm.update(cand)
    if not ok3:
        logging.info("[STAGE3] not_confirmed(%s) ts=%s side=%s strength=%s", reason3, bar_ts, side, strength)
        return

    # confirmed -> signal + decision
    logging.info("[STAGE3] CONFIRMED(%s) ts=%s side=%s strength=%s p_up=%.4f",
                 reason3, bar_ts, side, strength, p_up)

    upsert_signal(
        st.conn,
        symbol=st.symbol,
        ts=bar_ts,
        horizon=HORIZON_MIN,
        side=side,
        strength=strength,
        p_up=p_up,
        score=score,
        model=MODEL_NAME,
    )

    # ===== Decision Layer write point (核心) =====
    upsert_decision(
        st.conn,
        symbol=st.symbol,
        ts=bar_ts,
        horizon=HORIZON_MIN,
        side=side,
        strength=strength,
        p_up=p_up,
        score=score,
        model=MODEL_NAME,
        stage2_pass=1,
        stage3_pass=1,
        reason="ok",
        risk_pause_until=st.risk.paused_until_str(),
        status="OPEN",
    )

    logging.info("[DECISION] upserted ts=%s side=%s strength=%s model=%s", bar_ts, side, strength, MODEL_NAME)


def job_backfill_decisions(st: AppState):
    filled = backfill_decisions_outcome(st.conn, lookback_limit=2000)
    if filled:
        logging.info("[DECISION_BACKFILL] filled=%d", filled)


def job_monitor(st: AppState):
    mon = rolling_decisions_monitor(st.conn, model=MODEL_NAME, window=MONITOR_WINDOW)
    if mon.get("count", 0) >= 5:
        logging.info("[DECISION_MONITOR] %s", mon)


# =========================
# Runner
# =========================

def run_scheduler(symbol: str, use_demo: bool = True):
    st = AppState(symbol=symbol, use_demo=use_demo)

    sched = BackgroundScheduler()
    sched.add_job(lambda: job_collect_and_aggregate(st), "interval", seconds=COLLECT_INTERVAL_SEC, id="collect")
    sched.add_job(lambda: job_predict_and_decide(st), "interval", seconds=PREDICT_INTERVAL_SEC, id="predict")
    sched.add_job(lambda: job_backfill_decisions(st), "interval", seconds=BACKFILL_INTERVAL_SEC, id="backfill")
    sched.add_job(lambda: job_monitor(st), "interval", seconds=MONITOR_INTERVAL_SEC, id="monitor")

    sched.start()
    logging.info("[SCHED] started. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("[SCHED] shutting down...")
        sched.shutdown(wait=False)
        try:
            st.conn.close()
        except Exception:
            pass
        logging.info("[SCHED] stopped.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    ap.add_argument("--use_demo", type=int, default=1, help="1=use demo datasource, 0=placeholder for real datasource")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scheduler(symbol=args.symbol, use_demo=bool(args.use_demo))