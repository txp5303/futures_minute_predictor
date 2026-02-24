# stats_probability_bins.py
# ------------------------------------------------------------
# 概率分桶统计（signals_1m）
# 适配你的 signals_1m 表结构：
#   symbol, ts, side, strength, p_up, score, model,
#   last_close, future_close, realized_ret, hit_flag
#
# 关键增强：
# - 若 signals_1m 内 future_close / realized_ret / hit_flag 为空，
#   则自动去 kline_1m 查 next minute close 做回填计算（与 stats_signals.py 口径一致）
#
# 用法：
#   python stats_probability_bins.py --model schemeA+
#   python stats_probability_bins.py --model schemeA+_filtered_v1 --bins "0.65,0.70,0.75,0.80,0.85,0.90,1.01"
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def parse_bins(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("bins 至少需要 2 个边界，例如: 0.65,0.70,0.80,1.01")
    vals = [float(x) for x in parts]
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            raise ValueError("bins 必须严格递增，例如: 0.65,0.70,0.80,1.01")
    return vals


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def infer_db_path(cli_path: Optional[str]) -> str:
    if cli_path:
        return cli_path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "market.db")


def has_column(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table});")
    rows = cur.fetchall()
    return any(r[1] == col for r in rows)


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.4f},{hi:.4f})"


def assign_bin(p: float, edges: List[float]) -> Optional[int]:
    for i in range(len(edges) - 1):
        if edges[i] <= p < edges[i + 1]:
            return i
    return None


@dataclass
class Row:
    ts: str
    symbol: str
    side: str
    strength: str
    p_up: float
    score: Optional[float]
    model: str

    last_close: Optional[float]
    future_close: Optional[float]
    realized_ret: Optional[float]
    hit_flag: Optional[int]

    # 用于回填时查 kline_1m
    kline_close0: Optional[float] = None
    kline_close1: Optional[float] = None

    @property
    def filled(self) -> bool:
        # 任何一种有 next close 就算回填成功
        return (self.future_close is not None) or (self.hit_flag is not None) or (self.kline_close1 is not None)

    def get_close0(self) -> Optional[float]:
        return self.last_close if self.last_close is not None else self.kline_close0

    def get_close1(self) -> Optional[float]:
        return self.future_close if self.future_close is not None else self.kline_close1

    def get_realized_ret(self) -> Optional[float]:
        # 优先用表内 realized_ret
        if self.realized_ret is not None:
            return float(self.realized_ret)
        c0 = self.get_close0()
        c1 = self.get_close1()
        if c0 is None or c1 is None:
            return None
        # 与你 stats_signals.py 输出一致：用“点数差”而不是百分比
        return float(c1) - float(c0)

    def get_pnl(self) -> Optional[float]:
        r = self.get_realized_ret()
        if r is None:
            return None
        side = (self.side or "").upper()
        if side == "SHORT":
            return -r
        return r

    def get_hit_flag(self) -> Optional[int]:
        if self.hit_flag is not None:
            return int(self.hit_flag)
        pnl = self.get_pnl()
        if pnl is None:
            return None
        return 1 if pnl > 0 else 0


@dataclass
class Agg:
    total: int = 0
    filled: int = 0
    wins: int = 0
    sum_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss_abs: float = 0.0
    win_sum: float = 0.0
    win_cnt: int = 0
    loss_sum_abs: float = 0.0
    loss_cnt: int = 0

    def add(self, r: Row) -> None:
        self.total += 1
        if r.filled:
            self.filled += 1

        pnl = r.get_pnl()
        if pnl is None:
            return

        self.sum_pnl += pnl
        if pnl > 0:
            self.gross_profit += pnl
            self.win_sum += pnl
            self.win_cnt += 1
        elif pnl < 0:
            self.gross_loss_abs += abs(pnl)
            self.loss_sum_abs += abs(pnl)
            self.loss_cnt += 1

        hf = r.get_hit_flag()
        if hf is not None and hf == 1:
            self.wins += 1

    @property
    def winrate(self) -> Optional[float]:
        if self.filled == 0:
            return None
        return self.wins / self.filled

    @property
    def pf(self) -> Optional[float]:
        return safe_div(self.gross_profit, self.gross_loss_abs)

    @property
    def exp(self) -> Optional[float]:
        if self.total == 0:
            return None
        return self.sum_pnl / self.total

    @property
    def rr(self) -> Optional[float]:
        if self.win_cnt == 0 or self.loss_cnt == 0:
            return None
        avg_win = self.win_sum / self.win_cnt
        avg_loss = self.loss_sum_abs / self.loss_cnt
        return safe_div(avg_win, avg_loss)


def load_kline_close_map(cur: sqlite3.Cursor, symbol: str, ts_min: str, ts_max: str) -> Dict[str, float]:
    """
    从 kline_1m 读取 close，用于回填：
    - 需要 ts_min ~ ts_max 范围（包含）内的 close
    """
    if not has_column(cur, "kline_1m", "close"):
        raise RuntimeError("表 kline_1m 缺少 close 列，无法回填。")
    # 有的表列名为 last_close？但你前面的系统是 close
    sql = """
    SELECT ts, close
    FROM kline_1m
    WHERE symbol=? AND ts>=? AND ts<=?
    """
    cur.execute(sql, (symbol, ts_min, ts_max))
    mp: Dict[str, float] = {}
    for ts, close in cur.fetchall():
        if close is None:
            continue
        mp[str(ts)] = float(close)
    return mp


def main():
    ap = argparse.ArgumentParser(description="signals_1m 概率(p_up)分桶统计（自动从 kline_1m 回填）")
    ap.add_argument("--db", default=None, help="SQLite DB 路径（默认: ./data/market.db）")
    ap.add_argument("--table", default="signals_1m", help="表名（默认 signals_1m）")
    ap.add_argument("--symbol", default="螺纹", help="品种（默认 螺纹）")
    ap.add_argument("--model", default=None, help="按 model 精确过滤（默认不过滤）")
    ap.add_argument("--min_ts", default=None, help='起始时间（包含），格式 "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--max_ts", default=None, help='结束时间（包含），格式 "YYYY-MM-DD HH:MM:SS"')
    ap.add_argument("--filled_only", action="store_true", help="仅统计可回填/已回填数据（能拿到 next close）")
    ap.add_argument(
        "--bins",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,1.01",
        help='分桶边界，逗号分隔，严格递增。例: "0.65,0.70,0.80,1.01"',
    )
    ap.add_argument("--topn", type=int, default=10, help="显示最近 topN 条（默认 10）")
    args = ap.parse_args()

    db_path = infer_db_path(args.db)
    edges = parse_bins(args.bins)
    tbl = args.table

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB 不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 必要列检查（按你当前结构）
    required_cols = ["ts", "symbol", "side", "strength", "p_up", "model"]
    for c in required_cols:
        if not has_column(cur, tbl, c):
            raise RuntimeError(f"表 {tbl} 缺少必要列: {c}")

    # 可选列
    has_score = has_column(cur, tbl, "score")
    has_last_close = has_column(cur, tbl, "last_close")
    has_fclose = has_column(cur, tbl, "future_close")
    has_rret = has_column(cur, tbl, "realized_ret")
    has_hit = has_column(cur, tbl, "hit_flag")

    select_cols = ["ts", "symbol", "side", "strength", "p_up", "model"]
    select_cols.append("score" if has_score else "NULL AS score")
    select_cols.append("last_close" if has_last_close else "NULL AS last_close")
    select_cols.append("future_close" if has_fclose else "NULL AS future_close")
    select_cols.append("realized_ret" if has_rret else "NULL AS realized_ret")
    select_cols.append("hit_flag" if has_hit else "NULL AS hit_flag")

    where = ["symbol = ?"]
    params: List[object] = [args.symbol]

    if args.model:
        where.append("model = ?")
        params.append(args.model)

    if args.min_ts:
        where.append("ts >= ?")
        params.append(args.min_ts)
    if args.max_ts:
        where.append("ts <= ?")
        params.append(args.max_ts)

    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM {tbl}
    WHERE {" AND ".join(where)}
    ORDER BY ts ASC
    """
    cur.execute(sql, params)
    raw = cur.fetchall()

    if not raw:
        print("无数据：请检查 symbol/model/时间范围。")
        conn.close()
        return

    # 确定回填需要的 kline 范围：signals ts 的 min~max + 1min
    ts_list = [str(rr["ts"]) for rr in raw]
    ts_min = min(ts_list)
    ts_max = max(ts_list)
    dt_max = parse_ts(ts_max) + timedelta(minutes=1)
    ts_max_plus1 = format_ts(dt_max)

    # 读取 kline_1m close 映射
    kmap = load_kline_close_map(cur, args.symbol, ts_min, ts_max_plus1)

    rows: List[Row] = []
    for rr in raw:
        ts = str(rr["ts"])
        dt0 = parse_ts(ts)
        ts1 = format_ts(dt0 + timedelta(minutes=1))

        r = Row(
            ts=ts,
            symbol=str(rr["symbol"]),
            side=str(rr["side"]),
            strength=str(rr["strength"]),
            p_up=float(rr["p_up"]),
            score=(float(rr["score"]) if rr["score"] is not None else None),
            model=str(rr["model"]),
            last_close=(float(rr["last_close"]) if rr["last_close"] is not None else None),
            future_close=(float(rr["future_close"]) if rr["future_close"] is not None else None),
            realized_ret=(float(rr["realized_ret"]) if rr["realized_ret"] is not None else None),
            hit_flag=(int(rr["hit_flag"]) if rr["hit_flag"] is not None else None),
            kline_close0=(kmap.get(ts)),
            kline_close1=(kmap.get(ts1)),
        )
        rows.append(r)

    # 如果只统计可回填的（有 next close）
    if args.filled_only:
        rows = [r for r in rows if r.get_close1() is not None]
        if not rows:
            print("filled_only 下无可回填样本（可能 kline_1m 缺少对应分钟）。")
            conn.close()
            return

    # 分桶聚合
    bin_aggs: Dict[int, Agg] = {i: Agg() for i in range(len(edges) - 1)}
    bin_long: Dict[int, Agg] = {i: Agg() for i in range(len(edges) - 1)}
    bin_short: Dict[int, Agg] = {i: Agg() for i in range(len(edges) - 1)}
    dropped = 0

    overall = Agg()
    overall_long = Agg()
    overall_short = Agg()

    for r in rows:
        overall.add(r)
        if (r.side or "").upper() == "SHORT":
            overall_short.add(r)
        else:
            overall_long.add(r)

        bi = assign_bin(r.p_up, edges)
        if bi is None:
            dropped += 1
            continue
        bin_aggs[bi].add(r)
        if (r.side or "").upper() == "SHORT":
            bin_short[bi].add(r)
        else:
            bin_long[bi].add(r)

    # 输出
    print("\n==============================")
    print("   概率分桶统计（signals_1m）")
    print("==============================")
    print(f"数据库路径: {db_path}")
    print(f"table: {tbl}")
    print(f"symbol: {args.symbol}")
    print(f"model: {args.model if args.model else '(ALL models)'}")
    if args.min_ts or args.max_ts:
        print(f"ts_range: [{args.min_ts or '-inf'}, {args.max_ts or '+inf'}]")
    print(f"bins: {edges}")
    if args.filled_only:
        print("mode: filled_only=TRUE (基于 kline_1m 可回填)")
    print(f"总记录数: {len(rows)}  | 未落入任何桶(dropped): {dropped}")

    print("\n总体统计：")
    print(
        "  ALL : total={t} filled={f} winrate={wr} sum_pnl={sp} PF={pf} Exp={ex} RR={rr}".format(
            t=overall.total, f=overall.filled, wr=fmt(overall.winrate),
            sp=fmt(overall.sum_pnl), pf=fmt(overall.pf), ex=fmt(overall.exp), rr=fmt(overall.rr),
        )
    )
    print(
        "  LONG: total={t} filled={f} winrate={wr} sum_pnl={sp} PF={pf} Exp={ex} RR={rr}".format(
            t=overall_long.total, f=overall_long.filled, wr=fmt(overall_long.winrate),
            sp=fmt(overall_long.sum_pnl), pf=fmt(overall_long.pf), ex=fmt(overall_long.exp), rr=fmt(overall_long.rr),
        )
    )
    print(
        "  SHORT: total={t} filled={f} winrate={wr} sum_pnl={sp} PF={pf} Exp={ex} RR={rr}".format(
            t=overall_short.total, f=overall_short.filled, wr=fmt(overall_short.winrate),
            sp=fmt(overall_short.sum_pnl), pf=fmt(overall_short.pf), ex=fmt(overall_short.exp), rr=fmt(overall_short.rr),
        )
    )

    print("\n分桶统计（ALL / LONG / SHORT）：")
    header = (
        f"{'BIN':<16} | "
        f"{'ALL(t/f/wr/PF/Exp)':<34} | "
        f"{'LONG(t/f/wr/PF/Exp)':<34} | "
        f"{'SHORT(t/f/wr/PF/Exp)':<34}"
    )
    print(header)
    print("-" * len(header))

    def pack(x: Agg) -> str:
        return f"{x.total:>3}/{x.filled:<3} {fmt(x.winrate):>6} {fmt(x.pf):>6} {fmt(x.exp):>7}"

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lab = bin_label(lo, hi)
        print(
            f"{lab:<16} | "
            f"{pack(bin_aggs[i]):<34} | "
            f"{pack(bin_long[i]):<34} | "
            f"{pack(bin_short[i]):<34}"
        )

    # 最近 topN（按可回填优先）
    topn = max(0, int(args.topn))
    if topn > 0:
        filled_rows = [r for r in rows if r.filled]
        show_rows = filled_rows[-topn:] if len(filled_rows) >= topn else filled_rows

        print(f"\n最近{len(show_rows)}条（可回填优先）：")
        for r in show_rows:
            c0 = r.get_close0()
            c1 = r.get_close1()
            ret = r.get_realized_ret()
            pnl = r.get_pnl()
            hf = r.get_hit_flag()
            print(
                f"{r.ts} | model={r.model} | {r.side}/{r.strength} | "
                f"p_up={r.p_up:.4f} | score={fmt(r.score)} | "
                f"{fmt(c0)}->{fmt(c1)} | ret={fmt(ret)} | pnl={fmt(pnl)} | hit={hf}"
            )

    # 快速诊断序列
    exps = [bin_aggs[i].exp for i in range(len(edges) - 1)]
    pfs = [bin_aggs[i].pf for i in range(len(edges) - 1)]
    counts = [bin_aggs[i].total for i in range(len(edges) - 1)]

    print("\n快速诊断：")
    print(f"  per-bin counts: {counts}")
    print("  per-bin Exp   :", "[" + ", ".join(fmt(x, 4) for x in exps) + "]")
    print("  per-bin PF    :", "[" + ", ".join(fmt(x, 4) for x in pfs) + "]")
    print("  观察要点：p_up 越高的桶，Exp/PF 是否更好、更稳定？若不单调，考虑“趋势末端追高过滤/特征重构”。")

    conn.close()


if __name__ == "__main__":
    main()