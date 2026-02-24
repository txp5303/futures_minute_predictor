import argparse
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# Helpers
# ============================================================

def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def load_signals(conn: sqlite3.Connection, model: str, symbol: str) -> pd.DataFrame:
    """
    signals_1m schema (from user):
      symbol TEXT, ts TEXT, side TEXT, strength TEXT, p_up REAL, score REAL, model TEXT,
      last_close REAL, future_close REAL, realized_ret REAL, hit_flag INTEGER
    """
    if not _table_exists(conn, "signals_1m"):
        raise RuntimeError("DB 中不存在表 signals_1m")

    cols = _table_columns(conn, "signals_1m")
    need = ["symbol", "ts", "side", "strength", "p_up", "score", "model"]
    for c in need:
        if c not in cols:
            raise RuntimeError(f"signals_1m 缺少字段: {c}，实际字段={cols}")

    # optional cols
    has_last_close = "last_close" in cols
    has_future_close = "future_close" in cols
    has_realized_ret = "realized_ret" in cols
    has_hit_flag = "hit_flag" in cols

    select_cols = ["symbol", "ts", "side", "strength", "p_up", "score", "model"]
    if has_last_close:
        select_cols.append("last_close")
    if has_future_close:
        select_cols.append("future_close")
    if has_realized_ret:
        select_cols.append("realized_ret")
    if has_hit_flag:
        select_cols.append("hit_flag")

    sql = f"""
    SELECT {", ".join(select_cols)}
    FROM signals_1m
    WHERE model = ?
      AND symbol = ?
    ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(model, symbol))

    # normalize
    df["ts"] = pd.to_datetime(df["ts"])
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["strength"] = df["strength"].astype(str).str.upper().str.strip()
    df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # keep only plausible rows
    df = df.dropna(subset=["ts", "side", "strength", "p_up"])
    return df


def compute_pnl_row(row: pd.Series) -> Optional[float]:
    """
    优先使用 realized_ret(如果存在且非空)；
    否则用 future_close - last_close，并按 side 方向转成方向性 pnl。
    """
    side = str(row.get("side", "")).upper()

    realized_ret = row.get("realized_ret", np.nan)
    if pd.notna(realized_ret):
        # 注意：你现有 stats_signals.py 展示的 ret/pnl 看起来就是“方向性收益”
        # 这里直接用 realized_ret 作为 pnl
        return float(realized_ret)

    last_close = row.get("last_close", np.nan)
    future_close = row.get("future_close", np.nan)
    if pd.isna(last_close) or pd.isna(future_close):
        return None

    diff = float(future_close) - float(last_close)
    if side == "LONG":
        return diff
    if side == "SHORT":
        return -diff
    return None


def add_pnl_and_hit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pnl"] = df.apply(compute_pnl_row, axis=1)

    # hit：优先用 hit_flag，否则用 pnl>0
    pnl_hit_fallback = pd.Series(np.where(df["pnl"] > 0, 1, 0), index=df.index)

    if "hit_flag" in df.columns:
        hit_series = pd.to_numeric(df["hit_flag"], errors="coerce")
        hit_series = hit_series.where(hit_series.notna(), pnl_hit_fallback)
        df["hit"] = hit_series.astype(int)
    else:
        df["hit"] = pnl_hit_fallback.astype(int)

    df["filled"] = df["pnl"].notna().astype(int)
    return df
    df = df.copy()
    df["pnl"] = df.apply(compute_pnl_row, axis=1)
    # hit：优先用 hit_flag，否则用 pnl>0
    if "hit_flag" in df.columns:
        df["hit"] = pd.to_numeric(df["hit_flag"], errors="coerce")
        df["hit"] = df["hit"].fillna(np.where(df["pnl"] > 0, 1, 0)).astype(int)
    else:
        df["hit"] = np.where(df["pnl"] > 0, 1, 0).astype(int)

    df["filled"] = df["pnl"].notna().astype(int)
    return df


def calc_stats(df: pd.DataFrame) -> dict:
    """
    PF = sum(win pnl) / abs(sum(loss pnl))
    Exp = mean(pnl)
    """
    filled = df[df["pnl"].notna()].copy()
    total = int(len(df))
    filled_n = int(len(filled))
    if filled_n == 0:
        return dict(total=total, filled=0, hit=0, winrate=np.nan, sum_pnl=0.0, PF=np.nan, Exp=np.nan)

    hit = int((filled["pnl"] > 0).sum())
    winrate = hit / filled_n if filled_n else np.nan
    sum_pnl = float(filled["pnl"].sum())

    wins = filled.loc[filled["pnl"] > 0, "pnl"].sum()
    losses = filled.loc[filled["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        PF = np.inf if wins > 0 else np.nan
    else:
        PF = float(wins / abs(losses))

    Exp = float(filled["pnl"].mean())
    return dict(total=total, filled=filled_n, hit=hit, winrate=winrate, sum_pnl=sum_pnl, PF=PF, Exp=Exp)


def strength_rank(s: str) -> int:
    # L3 > L2 > L1 > other
    s = (s or "").upper()
    if s == "L3":
        return 3
    if s == "L2":
        return 2
    if s == "L1":
        return 1
    return 0


def apply_stage2_filter(
    df: pd.DataFrame,
    only_long: bool,
    min_strength: str,
    p_up_low: float,
    p_up_high: float,
) -> pd.DataFrame:
    """
    Stage2 核心过滤（与你现在日志里一致的“只做 LONG/L2+ + PUP区间”那类规则）
    - 方向：默认 ONLY LONG（可改为允许 SHORT）
    - 强度：>= min_strength（默认 L2）
    - 概率：p_up in (low, high]（默认 (0.57, 0.70] 之类）
    """
    out = df.copy()

    # side
    if only_long:
        out = out[out["side"] == "LONG"]
    else:
        out = out[out["side"].isin(["LONG", "SHORT"])]

    # strength
    min_r = strength_rank(min_strength)
    out = out[out["strength"].apply(strength_rank) >= min_r]

    # p_up interval: (low, high]
    out = out[(out["p_up"] > float(p_up_low)) & (out["p_up"] <= float(p_up_high))]

    return out


def print_block(title: str, df: pd.DataFrame):
    st = calc_stats(df)
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"total={st['total']} | filled={st['filled']} | hit={st['hit']} | winrate={st['winrate']:.4f}" if st["filled"] else f"total={st['total']} | filled=0")
    print(f"sum_pnl={st['sum_pnl']:.4f} | PF={st['PF'] if np.isfinite(st['PF']) else st['PF']} | Exp={st['Exp']:.4f}" if st["filled"] else "")

    if len(df) and "side" in df.columns:
        for side in ["LONG", "SHORT"]:
            sdf = df[df["side"] == side]
            if len(sdf) == 0:
                continue
            ss = calc_stats(sdf)
            print(f"  {side:5s}: total={ss['total']} filled={ss['filled']} winrate={ss['winrate'] if ss['filled'] else np.nan} sum_pnl={ss['sum_pnl']:.4f} PF={ss['PF']} Exp={ss['Exp']}")

    # by strength
    if len(df) and "strength" in df.columns:
        for lv in ["L1", "L2", "L3"]:
            sdf = df[df["strength"] == lv]
            if len(sdf) == 0:
                continue
            ss = calc_stats(sdf)
            print(f"  {lv:2s}: total={ss['total']} filled={ss['filled']} winrate={ss['winrate'] if ss['filled'] else np.nan} sum_pnl={ss['sum_pnl']:.4f} PF={ss['PF']} Exp={ss['Exp']}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market.db")
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default="螺纹")

    # Stage2 knobs (match your current usage)
    ap.add_argument("--min-strength", default="L2", choices=["L1", "L2", "L3"])
    ap.add_argument("--pup-low", type=float, default=0.57)
    ap.add_argument("--pup-high", type=float, default=0.70)

    # output controls
    ap.add_argument("--show-last", type=int, default=15, help="打印最近N条（过滤后）")

    args = ap.parse_args()

    print("=" * 60)
    print("Stage2 Shadow Backtest (signals_1m)")
    print("=" * 60)
    print(f"DB    : {args.db}")
    print(f"Model : {args.model}")
    print(f"Symbol: {args.symbol}")
    print(f"Stage2: min_strength>={args.min_strength}, PUP=({args.pup_low},{args.pup_high}]")
    print("-" * 60)

    conn = sqlite3.connect(args.db)

    try:
        sig = load_signals(conn, args.model, args.symbol)
    finally:
        conn.close()

    sig = add_pnl_and_hit(sig)

    # baseline: raw signals for this model
    print_block("RAW (all signals of this model)", sig)

    # Stage2 default (ONLY LONG)
    s2_long = apply_stage2_filter(
        sig,
        only_long=True,
        min_strength=args.min_strength,
        p_up_low=args.pup_low,
        p_up_high=args.pup_high,
    )
    print_block("STAGE2 (ONLY LONG) filtered", s2_long)

    # Shadow: allow SHORT too
    s2_both = apply_stage2_filter(
        sig,
        only_long=False,
        min_strength=args.min_strength,
        p_up_low=args.pup_low,
        p_up_high=args.pup_high,
    )
    print_block("STAGE2 SHADOW (ALLOW SHORT) filtered", s2_both)

    # delta summary
    stL = calc_stats(s2_long)
    stB = calc_stats(s2_both)
    print("\n" + "-" * 60)
    print("DELTA (ALLOW SHORT minus ONLY LONG)")
    if stL["filled"] == 0 and stB["filled"] == 0:
        print("两者都没有可回填记录。")
    else:
        print(f"filled: {stB['filled']} - {stL['filled']} = {stB['filled'] - stL['filled']}")
        print(f"sum_pnl: {stB['sum_pnl']:.4f} - {stL['sum_pnl']:.4f} = {(stB['sum_pnl'] - stL['sum_pnl']):.4f}")
        if stL["filled"] and stB["filled"]:
            print(f"Exp: {stB['Exp']:.4f} - {stL['Exp']:.4f} = {(stB['Exp'] - stL['Exp']):.4f}")
            print(f"winrate: {stB['winrate']:.4f} - {stL['winrate']:.4f} = {(stB['winrate'] - stL['winrate']):.4f}")

    # print last N rows for quick inspection
    if args.show_last > 0:
        tail = s2_both.tail(args.show_last).copy()
        if len(tail):
            print("\n" + "=" * 60)
            print(f"LAST {min(args.show_last, len(tail))} rows (STAGE2 SHADOW filtered)")
            print("=" * 60)
            show_cols = [c for c in ["ts", "side", "strength", "p_up", "score", "pnl", "hit"] if c in tail.columns]
            tail = tail.sort_values("ts")
            for _, r in tail.iterrows():
                ts = r["ts"].strftime("%Y-%m-%d %H:%M:%S")
                side = r.get("side", "")
                strength = r.get("strength", "")
                p_up = float(r.get("p_up", np.nan))
                score = float(r.get("score", np.nan))
                pnl = r.get("pnl", np.nan)
                hit = int(r.get("hit", 0))
                print(f"{ts} | {side}/{strength} | p_up={p_up:.4f} score={score:.6f} | pnl={pnl if pd.notna(pnl) else 'NA'} | hit={hit}")

    print("\nDone.")


if __name__ == "__main__":
    main()