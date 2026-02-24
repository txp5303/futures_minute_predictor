# stats_stage2_grid.py
# ------------------------------------------------------------
# Stage2 参数网格统计（基于 signals_1m + kline_1m 计算 vol/jump 特征）
# 你可以直接复制覆盖整个文件后运行：
#   python stats_stage2_grid.py --model schemeA+_filtered_v1 --pup_lo_list "0.57" --pup_hi_list "0.70" --vol_list "6.5" --jump_list "12"
#
# 说明：
# - signals_1m 表结构（你已贴过）：
#   symbol, ts, side, strength, p_up, score, model, last_close, future_close, realized_ret, hit_flag
# - 本脚本会从 kline_1m 取 close 序列，按 rolling 计算：
#   vol_mean_abs = rolling(mean(abs(diff(close))), window=--vol_window)
#   jump_abs     = abs(diff(close))（即当根K线的跳变）
# - Stage2 过滤默认：ONLY LONG + min_strength>=L2 + p_up 区间 + vol/jump 阈值
# - 可加 --allow_short 观察“允许 SHORT”的网格结果（仍以 p_up 区间过滤）
# ------------------------------------------------------------

import argparse
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config / Utils
# -----------------------------

STRENGTH_RANK = {"L1": 1, "L2": 2, "L3": 3}


def parse_csv_floats(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return [float(x) for x in parts]


def parse_strength(s: str) -> str:
    s = (s or "").strip().upper()
    if s not in STRENGTH_RANK:
        raise ValueError(f"Invalid strength: {s}. Use L1/L2/L3.")
    return s


def strength_ge(strength: str, min_strength: str) -> bool:
    return STRENGTH_RANK.get(strength, 0) >= STRENGTH_RANK[min_strength]


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return np.nan
    return a / b


def compute_pf(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    if pnl.empty:
        return np.nan
    pos = pnl[pnl > 0].sum()
    neg = (-pnl[pnl < 0]).sum()
    if neg == 0:
        return np.inf if pos > 0 else np.nan
    return float(pos / neg)


def compute_winrate(hit: pd.Series) -> float:
    hit = pd.to_numeric(hit, errors="coerce").dropna()
    if hit.empty:
        return np.nan
    return float(hit.mean())


def ensure_pnl_hit(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复点：
    - 不再对 fillna() 传 ndarray（会触发你遇到的 TypeError）
    - fallback_hit 用 Series(index=df.index)
    """
    df = df.copy()

    # pnl 优先 realized_ret
    if "realized_ret" in df.columns and df["realized_ret"].notna().any():
        df["pnl"] = pd.to_numeric(df["realized_ret"], errors="coerce")
    else:
        lc = pd.to_numeric(df.get("last_close"), errors="coerce")
        fc = pd.to_numeric(df.get("future_close"), errors="coerce")
        raw = fc - lc
        side = df.get("side", "").astype(str).str.upper()
        df["pnl"] = pd.Series(np.where(side == "SHORT", -raw, raw), index=df.index)

    # hit：优先 hit_flag，否则 pnl>0
    fallback_hit = pd.Series(np.where(df["pnl"] > 0, 1, 0), index=df.index)

    if "hit_flag" in df.columns:
        hit_series = pd.to_numeric(df["hit_flag"], errors="coerce")
        hit_series = hit_series.where(hit_series.notna(), fallback_hit)
        df["hit"] = hit_series.astype(int)
    else:
        df["hit"] = fallback_hit.astype(int)

    return df


# -----------------------------
# DB Load
# -----------------------------

def table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols


def load_signals(conn: sqlite3.Connection, model: str, symbol: str) -> pd.DataFrame:
    sql = """
    SELECT
        symbol, ts, side, strength, p_up, score, model,
        last_close, future_close, realized_ret, hit_flag
    FROM signals_1m
    WHERE model = ?
      AND symbol = ?
    ORDER BY ts
    """
    df = pd.read_sql(sql, conn, params=(model, symbol))
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df["side"] = df["side"].astype(str).str.upper()
    df["strength"] = df["strength"].astype(str).str.upper()
    df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def load_kline_close(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """
    尽量兼容不同 kline_1m 表结构，只要有 ts + close 即可。
    """
    if not table_has_column(conn, "kline_1m", "close"):
        raise RuntimeError("kline_1m 表缺少 close 列，无法计算 vol/jump。")

    sql = """
    SELECT symbol, ts, close
    FROM kline_1m
    WHERE symbol = ?
    ORDER BY ts
    """
    k = pd.read_sql(sql, conn, params=(symbol,))
    if k.empty:
        return k
    k["ts"] = pd.to_datetime(k["ts"])
    k["close"] = pd.to_numeric(k["close"], errors="coerce")
    k = k.dropna(subset=["ts", "close"])
    return k


def add_vol_jump_features(sig: pd.DataFrame, kline: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    为每个 signal(ts) 加上：
      - jump_abs: abs(close_t - close_{t-1})
      - vol_mean_abs: rolling mean(abs(diff(close))), window=vol_window
    """
    if sig.empty:
        return sig.copy()

    if kline.empty:
        out = sig.copy()
        out["jump_abs"] = np.nan
        out["vol_mean_abs"] = np.nan
        return out

    k = kline.copy().sort_values("ts")
    k["diff"] = k["close"].diff()
    k["jump_abs"] = k["diff"].abs()
    k["vol_mean_abs"] = k["jump_abs"].rolling(window=vol_window, min_periods=vol_window).mean()

    feat = k[["ts", "jump_abs", "vol_mean_abs"]].copy()
    feat = feat.drop_duplicates("ts", keep="last")

    out = sig.merge(feat, on="ts", how="left")
    return out


# -----------------------------
# Stage2 filter + stats
# -----------------------------

@dataclass(frozen=True)
class GridPoint:
    pup_lo: float
    pup_hi: float
    vol_th: float
    jump_th: float


def stage2_filter(
    df: pd.DataFrame,
    *,
    min_strength: str,
    pup_lo: float,
    pup_hi: float,
    vol_th: float,
    jump_th: float,
    allow_short: bool,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    x = df.copy()

    # strength
    x = x[x["strength"].apply(lambda s: strength_ge(str(s).upper(), min_strength))]

    # side
    if not allow_short:
        x = x[x["side"] == "LONG"]

    # p_up 区间（按你的 Stage2： (lo, hi]）
    x = x[(x["p_up"] > pup_lo) & (x["p_up"] <= pup_hi)]

    # vol/jump（可能 NaN：rolling 未满窗口时会 NaN，这里直接剔除）
    x = x[pd.to_numeric(x["vol_mean_abs"], errors="coerce").notna()]
    x = x[pd.to_numeric(x["jump_abs"], errors="coerce").notna()]

    x = x[x["vol_mean_abs"] <= vol_th]
    x = x[x["jump_abs"] <= jump_th]

    return x


def summarize(df: pd.DataFrame) -> Tuple[int, int, float, float, float]:
    """
    returns: (total, filled, winrate, sum_pnl, exp)
    这里 filled 就按 df 行数（你的 signals_1m 已经是可回填优先的记录；pnl/hit 也都能算）
    """
    if df.empty:
        return 0, 0, np.nan, 0.0, np.nan
    df = ensure_pnl_hit(df)
    pnl = df["pnl"]
    hit = df["hit"]
    total = len(df)
    filled = len(df)
    winrate = compute_winrate(hit)
    sum_pnl = float(pd.to_numeric(pnl, errors="coerce").sum())
    exp = float(pd.to_numeric(pnl, errors="coerce").mean())
    return total, filled, winrate, sum_pnl, exp


def summarize_pf(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    df = ensure_pnl_hit(df)
    return compute_pf(df["pnl"])


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/market.db", help="sqlite db path (default: data/market.db)")
    ap.add_argument("--model", required=True, help="model name in signals_1m.model")
    ap.add_argument("--symbol", default="螺纹", help="symbol (default: 螺纹)")

    ap.add_argument("--min_strength", default="L2", type=parse_strength, help="L1/L2/L3 (default: L2)")
    ap.add_argument("--allow_short", action="store_true", help="if set, allow SHORT in filtering")

    ap.add_argument("--pup_lo_list", required=True, help='csv floats, e.g. "0.57,0.60"')
    ap.add_argument("--pup_hi_list", required=True, help='csv floats, e.g. "0.70,0.75"')
    ap.add_argument("--vol_list", required=True, help='csv floats, e.g. "6.5,8,10"')
    ap.add_argument("--jump_list", required=True, help='csv floats, e.g. "12,15,20"')

    ap.add_argument("--vol_window", type=int, default=20, help="rolling window for vol_mean_abs (default: 20)")
    ap.add_argument("--top", type=int, default=30, help="print top N rows (default: 30)")

    args = ap.parse_args()

    pup_los = parse_csv_floats(args.pup_lo_list)
    pup_his = parse_csv_floats(args.pup_hi_list)
    vols = parse_csv_floats(args.vol_list)
    jumps = parse_csv_floats(args.jump_list)

    if not pup_los or not pup_his or not vols or not jumps:
        raise SystemExit("pup_lo_list / pup_hi_list / vol_list / jump_list must be non-empty.")

    print("=" * 110)
    print("Stage2 Grid Backtest (signals_1m + kline_1m features)")
    print("=" * 110)
    print(f"DB        : {args.db}")
    print(f"Model     : {args.model}")
    print(f"Symbol    : {args.symbol}")
    print(f"min_strength >= {args.min_strength} | allow_short={args.allow_short}")
    print(f"vol_window: {args.vol_window}")
    print("-" * 110)

    conn = sqlite3.connect(args.db)
    try:
        sig = load_signals(conn, args.model, args.symbol)
        if sig.empty:
            print("No signals found for this model/symbol.")
            return

        # add pnl/hit first (raw stats)
        sig = ensure_pnl_hit(sig)

        # add vol/jump features
        try:
            kline = load_kline_close(conn, args.symbol)
            sig = add_vol_jump_features(sig, kline, args.vol_window)
        except Exception as e:
            print(f"[WARN] cannot compute vol/jump from kline_1m: {e}")
            sig["jump_abs"] = np.nan
            sig["vol_mean_abs"] = np.nan

        # RAW baseline
        raw_total, raw_filled, raw_wr, raw_sum, raw_exp = summarize(sig)
        raw_pf = summarize_pf(sig)
        print("RAW (all signals of this model)")
        print(f"total={raw_total} | filled={raw_filled} | winrate={raw_wr:.4f} | sum_pnl={raw_sum:.4f} | PF={raw_pf} | Exp={raw_exp:.4f}")
        print("-" * 110)

        rows = []
        for lo in pup_los:
            for hi in pup_his:
                if hi <= lo:
                    continue
                for v in vols:
                    for j in jumps:
                        gp = GridPoint(pup_lo=lo, pup_hi=hi, vol_th=v, jump_th=j)
                        f = stage2_filter(
                            sig,
                            min_strength=args.min_strength,
                            pup_lo=gp.pup_lo,
                            pup_hi=gp.pup_hi,
                            vol_th=gp.vol_th,
                            jump_th=gp.jump_th,
                            allow_short=args.allow_short,
                        )
                        total, filled, wr, sum_pnl, exp = summarize(f)
                        pf = summarize_pf(f)
                        rows.append(
                            {
                                "pup_lo": gp.pup_lo,
                                "pup_hi": gp.pup_hi,
                                "vol_th": gp.vol_th,
                                "jump_th": gp.jump_th,
                                "total": total,
                                "filled": filled,
                                "winrate": wr,
                                "sum_pnl": sum_pnl,
                                "PF": pf,
                                "Exp": exp,
                            }
                        )

        res = pd.DataFrame(rows)
        if res.empty:
            print("No grid results (check lists).")
            return

        # 清理一下：没有样本的 Exp/PF 可能 NaN
        res["Exp_rank"] = res["Exp"].fillna(-1e18)
        res = res.sort_values(by=["Exp_rank", "PF", "filled"], ascending=[False, False, False]).drop(columns=["Exp_rank"])

        topn = int(args.top)
        print(f"TOP {topn} grid points (sorted by Exp desc, then PF, then filled)")
        print("-" * 110)
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.width", 200)
        print(res.head(topn).to_string(index=False))
        print("-" * 110)

        # 额外：把最优点对应的最近若干条明细也打印出来，方便你看是否“追高/极端波动”造成亏损
        best = res.head(1).iloc[0].to_dict()
        best_f = stage2_filter(
            sig,
            min_strength=args.min_strength,
            pup_lo=float(best["pup_lo"]),
            pup_hi=float(best["pup_hi"]),
            vol_th=float(best["vol_th"]),
            jump_th=float(best["jump_th"]),
            allow_short=args.allow_short,
        )
        best_f = ensure_pnl_hit(best_f).sort_values("ts")
        print("BEST grid point:")
        print(
            f"pup=({best['pup_lo']},{best['pup_hi']}], "
            f"vol_mean_abs<={best['vol_th']}, jump_abs<={best['jump_th']} | "
            f"filled={int(best['filled'])} Exp={best['Exp']}"
        )
        print("-" * 110)

        tail = best_f.tail(10)
        if tail.empty:
            print("No rows under BEST grid point.")
        else:
            for _, r in tail.iterrows():
                ts = r["ts"].strftime("%Y-%m-%d %H:%M:%S")
                side = str(r["side"])
                strength = str(r["strength"])
                p_up = float(r["p_up"]) if pd.notna(r["p_up"]) else np.nan
                score = float(r["score"]) if pd.notna(r["score"]) else np.nan
                pnl = float(r["pnl"]) if pd.notna(r["pnl"]) else np.nan
                hit = int(r["hit"]) if pd.notna(r["hit"]) else -1
                vma = float(r["vol_mean_abs"]) if pd.notna(r["vol_mean_abs"]) else np.nan
                ja = float(r["jump_abs"]) if pd.notna(r["jump_abs"]) else np.nan
                print(
                    f"{ts} | {side}/{strength} | p_up={p_up:.4f} score={score:.6f} | "
                    f"vol_mean_abs={vma:.4f} jump_abs={ja:.4f} | pnl={pnl:.4f} hit={hit}"
                )

        print("\nDone.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()