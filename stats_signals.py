# -*- coding: utf-8 -*-
import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "market.db"

# ✅ 阶段二：默认锁死统计对象
DEFAULT_MODEL = "schemeA+"


def table_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cols = set()
    for r in cur.execute(f"PRAGMA table_info({table})").fetchall():
        cols.add(r[1])
    return cols


def parse_bins(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(float(p))
    out = sorted(set(out))
    return out


def bin_label(x: float, bins: list[float]) -> str:
    # bins like [0.40,0.43,...,0.60]
    if x <= bins[0]:
        return f"(-inf,{bins[0]:.2f}]"
    for i in range(1, len(bins)):
        lo = bins[i - 1]
        hi = bins[i]
        if lo < x <= hi:
            return f"({lo:.2f},{hi:.2f}]"
    return f"({bins[-1]:.2f},+inf)"


def compute_stats(rows: list[dict]) -> dict:
    total = len(rows)
    filled = total  # 这里 rows 传入的本身就是“可回填样本”
    hit = sum(1 for r in rows if r["hit"] == 1)
    winrate = (hit / filled) if filled else 0.0

    pnls = [r["pnl"] for r in rows]
    sum_pnl = sum(pnls) if pnls else 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    sum_win = sum(wins) if wins else 0.0
    sum_loss = sum(losses) if losses else 0.0  # negative
    pf = (sum_win / abs(sum_loss)) if sum_loss < 0 else (float("inf") if sum_win > 0 else 0.0)

    avg_win = (sum_win / len(wins)) if wins else 0.0
    avg_loss = (sum_loss / len(losses)) if losses else 0.0  # negative
    exp = (sum_pnl / filled) if filled else 0.0
    rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else (float("inf") if avg_win > 0 else 0.0)

    return {
        "total": total,
        "filled": filled,
        "hit": hit,
        "winrate": winrate,
        "sum_pnl": sum_pnl,
        "pf": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "exp": exp,
        "rr": rr,
    }


def main():
    ap = argparse.ArgumentParser(description="Signals stats (signals_1m)")

    ap.add_argument("--db", default=str(DB_PATH), help="db path")
    ap.add_argument("--symbol", default="螺纹", help="symbol")
    ap.add_argument("--horizon", type=int, default=1, help="minutes horizon for backfill")

    # ✅ 阶段二：默认只统计 schemeA+
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"filter model (default locked to {DEFAULT_MODEL}); if column not exists, will ignore",
    )
    ap.add_argument(
        "--no-model-filter",
        action="store_true",
        help="disable model filter even if model column exists (NOT recommended for Stage2)",
    )

    ap.add_argument("--after", default=None, help="filter ts>=... (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--where", default=None, help="extra SQL where clause (without WHERE)")
    ap.add_argument("--last", type=int, default=10, help="print last N filled rows")

    ap.add_argument(
        "--pup-bins",
        default=None,
        help="comma floats, e.g. 0.40,0.43,0.45,0.47,0.50,0.53,0.55,0.57,0.60",
    )
    ap.add_argument("--pup-bin-strength", default=None, help="only bin stats for this strength, e.g. L3")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print("数据库不存在，请先运行 main.py 生成数据。")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sig_cols = table_columns(cur, "signals_1m")
    k_cols = table_columns(cur, "kline_1m")

    # 基本字段检查
    need_sig = {"ts", "symbol", "side", "strength", "p_up", "score", "last_close"}
    if not need_sig.issubset(sig_cols):
        print("signals_1m 缺少必要列：", sorted(need_sig - sig_cols))
        conn.close()
        return
    if not {"ts", "symbol", "close"}.issubset(k_cols):
        print("kline_1m 缺少必要列：", sorted({"ts", "symbol", "close"} - k_cols))
        conn.close()
        return

    wh = ["symbol=?"]
    params = [args.symbol]

    # ✅ model 过滤（若 signals_1m 有 model 列且未关闭过滤）
    model_filter_applied = False
    if (not args.no_model_filter) and args.model and ("model" in sig_cols):
        wh.append("model=?")
        params.append(args.model)
        model_filter_applied = True
    elif (not args.no_model_filter) and args.model and ("model" not in sig_cols):
        # 用户指定了 model，但表里没有该列：提示并忽略
        print(f"[WARN] signals_1m 无 model 列，无法按 model={args.model} 过滤，已忽略。")
    elif args.no_model_filter:
        print("[WARN] 你启用了 --no-model-filter（将统计所有 model 的信号），阶段二不推荐。")

    if args.after:
        wh.append("ts>=?")
        params.append(args.after)
    if args.where:
        wh.append(f"({args.where})")

    where_sql = "WHERE " + " AND ".join(wh) if wh else ""

    # 先取 signals
    select_cols = ["ts", "symbol", "side", "strength", "p_up", "score", "last_close"]
    if "model" in sig_cols:
        select_cols.append("model")

    sel = ", ".join(select_cols)
    cur.execute(f"SELECT {sel} FROM signals_1m {where_sql} ORDER BY ts ASC", params)
    sig_rows = [dict(zip(select_cols, r)) for r in cur.fetchall()]
    total = len(sig_rows)

    # 回填：找 ts+horizon 的k线 close
    filled_rows = []
    missing = 0

    for r in sig_rows:
        ts_str = r["ts"]
        try:
            dt0 = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            missing += 1
            continue

        dt1 = dt0 + timedelta(minutes=args.horizon)
        ts1 = dt1.strftime("%Y-%m-%d %H:%M:%S")

        row = cur.execute(
            "SELECT close FROM kline_1m WHERE symbol=? AND ts=?",
            (args.symbol, ts1),
        ).fetchone()
        if not row:
            missing += 1
            continue

        future_close = float(row[0])
        last_close = float(r["last_close"])
        side = str(r["side"]).upper()

        pnl = (future_close - last_close) if side == "LONG" else (last_close - future_close)
        hit = 1 if pnl > 0 else 0

        r2 = dict(r)
        r2["future_close"] = future_close
        r2["pnl"] = float(pnl)
        r2["hit"] = int(hit)
        filled_rows.append(r2)

    # 全部统计
    all_stats = compute_stats(filled_rows)

    # 方向分组
    long_rows = [r for r in filled_rows if str(r["side"]).upper() == "LONG"]
    short_rows = [r for r in filled_rows if str(r["side"]).upper() == "SHORT"]
    long_stats = compute_stats(long_rows)
    short_stats = compute_stats(short_rows)

    # 强度分组
    strengths = ["L1", "L2", "L3"]
    strength_stats = {}
    for s in strengths:
        s_rows = [r for r in filled_rows if str(r["strength"]).upper() == s]
        strength_stats[s] = compute_stats(s_rows)

    # 强度×方向
    cross_stats = {}
    for s in strengths:
        for d in ["LONG", "SHORT"]:
            key = f"{s}/{d}"
            cr = [r for r in filled_rows if str(r["strength"]).upper() == s and str(r["side"]).upper() == d]
            cross_stats[key] = compute_stats(cr)

    # 输出
    print("\n==============================")
    print("   过滤信号统计结果（signals_1m）")
    print("==============================")
    print(f"数据库路径: {db_path}")
    print(f"symbol: {args.symbol}")
    print(f"horizon: {args.horizon} minute(s)")

    if args.model and not args.no_model_filter:
        if model_filter_applied:
            print(f"model: {args.model} (已按 model 过滤 ✅)")
        else:
            print(f"model: {args.model} (想过滤但表无 model 列，已忽略 ⚠️)")
    elif args.no_model_filter:
        print("model: (未过滤，统计所有 model ⚠️)")

    if args.after:
        print(f"after: {args.after}")
    if args.where:
        print(f"where: {args.where}")
    print("")

    print(f"总信号条数: {total}")
    print(f"已完成回填: {len(filled_rows)}")
    print(f"缺少K线无法回填: {missing}")
    print(f"命中条数: {all_stats['hit']}")
    print(f"胜率(命中/已回填): {all_stats['winrate']:.4f}\n")

    print("总体收益统计（按方向pnl）:")
    print(
        f"  ALL: total={len(filled_rows)} filled={len(filled_rows)} hit={all_stats['hit']} "
        f"winrate={all_stats['winrate']:.4f} sum_pnl={all_stats['sum_pnl']:.4f} "
        f"PF={all_stats['pf']:.4f} Exp={all_stats['exp']:.4f} RR={all_stats['rr']:.4f}"
    )

    print("\n按方向分组统计：")
    print(
        f"  LONG: total={len(long_rows)} filled={len(long_rows)} hit={long_stats['hit']} "
        f"winrate={long_stats['winrate']:.4f} sum_pnl={long_stats['sum_pnl']:.4f} "
        f"PF={long_stats['pf']:.4f} Exp={long_stats['exp']:.4f} RR={long_stats['rr']:.4f}"
    )
    print(
        f"  SHORT: total={len(short_rows)} filled={len(short_rows)} hit={short_stats['hit']} "
        f"winrate={short_stats['winrate']:.4f} sum_pnl={short_stats['sum_pnl']:.4f} "
        f"PF={short_stats['pf']:.4f} Exp={short_stats['exp']:.4f} RR={short_stats['rr']:.4f}"
    )

    print("\n按强度分组统计：")
    for s in strengths:
        st = strength_stats[s]
        print(
            f"  {s}: total={st['total']} filled={st['filled']} hit={st['hit']} "
            f"winrate={st['winrate']:.4f} sum_pnl={st['sum_pnl']:.4f} "
            f"PF={st['pf']:.4f} Exp={st['exp']:.4f} RR={st['rr']:.4f}"
        )

    print("\n按强度×方向交叉统计：")
    for s in strengths:
        for d in ["LONG", "SHORT"]:
            k = f"{s}/{d}"
            st = cross_stats[k]
            print(
                f"  {k}: total={st['total']} filled={st['filled']} hit={st['hit']} "
                f"winrate={st['winrate']:.4f} sum_pnl={st['sum_pnl']:.4f} "
                f"PF={st['pf']:.4f} Exp={st['exp']:.4f} RR={st['rr']:.4f}"
            )

    # p_up 分桶（可选）
    if args.pup_bins:
        bins = parse_bins(args.pup_bins)
        strength_only = args.pup_bin_strength.upper() if args.pup_bin_strength else None

        print("\np_up 分桶统计：")
        if strength_only:
            print(f"  (仅 strength={strength_only}) bins={args.pup_bins}")
            base_rows = [r for r in filled_rows if str(r["strength"]).upper() == strength_only]
        else:
            print(f"  bins={args.pup_bins}")
            base_rows = list(filled_rows)

        buckets = {}
        for r in base_rows:
            pu = float(r["p_up"])
            lab = bin_label(pu, bins)
            buckets.setdefault(lab, []).append(r)

        order = [f"(-inf,{bins[0]:.2f}]"]
        for i in range(1, len(bins)):
            order.append(f"({bins[i-1]:.2f},{bins[i]:.2f}]")
        order.append(f"({bins[-1]:.2f},+inf)")

        for lab in order:
            rs = buckets.get(lab, [])
            st = compute_stats(rs)
            print(
                f"  {lab}: total={st['total']} filled={st['filled']} hit={st['hit']} "
                f"winrate={st['winrate']:.4f} sum_pnl={st['sum_pnl']:.4f} "
                f"PF={st['pf']:.4f} Exp={st['exp']:.4f} RR={st['rr']:.4f}"
            )

    # 最近N条（已回填）
    print(f"\n最近{args.last}条（已完成回填）：\n")
    if not filled_rows:
        print("暂无可回填数据，请再跑一会儿 main.py。")
    else:
        tail = filled_rows[-args.last:]
        tail = sorted(tail, key=lambda x: x["ts"], reverse=True)
        for r in tail:
            ts = r["ts"]
            side = r["side"]
            strength = r["strength"]
            p_up = float(r["p_up"])
            score = float(r["score"])
            lc = float(r["last_close"])
            fc = float(r["future_close"])
            pnl = float(r["pnl"])
            hit = int(r["hit"])

            model_str = ""
            if "model" in r:
                model_str = f" | model={r['model']}"

            print(
                f"{ts}{model_str} | {side}/{strength} | p_up={p_up:.4f} | score={score:.6f} | "
                f"{lc:.2f}->{fc:.2f} | ret={pnl:.5f} | hit={hit}"
            )

    conn.close()


if __name__ == "__main__":
    main()