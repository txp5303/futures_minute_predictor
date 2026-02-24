# -*- coding: utf-8 -*-
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "market.db"


def table_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cols = set()
    for r in cur.execute(f"PRAGMA table_info({table})").fetchall():
        cols.add(r[1])
    return cols


def build_where(symbol: str | None, model: str | None, after: str | None, extra_where: str | None, cols: set[str]) -> tuple[str, list]:
    wh = []
    params = []
    if symbol and "symbol" in cols:
        wh.append("symbol=?")
        params.append(symbol)
    if model and "model" in cols:
        wh.append("model=?")
        params.append(model)
    if after and "ts" in cols:
        # after 格式：YYYY-MM-DD HH:MM:SS
        wh.append("ts>=?")
        params.append(after)
    if extra_where:
        wh.append(f"({extra_where})")
    where_sql = ("WHERE " + " AND ".join(wh)) if wh else ""
    return where_sql, params


def main():
    ap = argparse.ArgumentParser(description="Predictions stats (predictions_1m)")
    ap.add_argument("--db", default=str(DB_PATH), help="db path")
    ap.add_argument("--symbol", default=None, help="filter symbol (e.g. 螺纹)")
    ap.add_argument("--model", default=None, help="filter model (if column exists)")
    ap.add_argument("--after", default=None, help="filter ts>=... (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--where", default=None, help="extra SQL where clause (without WHERE)")
    ap.add_argument("--last", type=int, default=10, help="print last N filled rows")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print("数据库不存在，请先运行 main.py 生成数据。")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cols = table_columns(cur, "predictions_1m")

    # 兼容：不同版本的回填字段命名可能不同
    # 优先使用：future_close / hit_flag / realized_ret
    has_future = "future_close" in cols
    has_hit = "hit_flag" in cols
    has_ret = "realized_ret" in cols

    where_sql, params = build_where(args.symbol, args.model, args.after, args.where, cols)

    cur.execute(f"SELECT COUNT(*) FROM predictions_1m {where_sql}", params)
    total = cur.fetchone()[0]

    filled = hit = 0
    winrate = 0.0

    if has_future:
        cur.execute(f"SELECT COUNT(*) FROM predictions_1m {where_sql} AND future_close IS NOT NULL" if where_sql else
                    "SELECT COUNT(*) FROM predictions_1m WHERE future_close IS NOT NULL", params)
        filled = cur.fetchone()[0]
        if has_hit:
            cur.execute(f"SELECT COUNT(*) FROM predictions_1m {where_sql} AND hit_flag=1" if where_sql else
                        "SELECT COUNT(*) FROM predictions_1m WHERE hit_flag=1", params)
            hit = cur.fetchone()[0]
            winrate = (hit / filled) if filled else 0.0

    print("\n==============================")
    print("   在线预测统计结果（predictions_1m）")
    print("==============================")
    print(f"数据库路径: {db_path}")
    if args.symbol:
        print(f"symbol: {args.symbol}")
    if args.model:
        print(f"model: {args.model} (若表无该列则忽略)")
    if args.after:
        print(f"after: {args.after}")
    if args.where:
        print(f"where: {args.where}")
    print(f"总预测条数: {total}")

    if not has_future:
        print("提示：predictions_1m 没有 future_close 列，无法计算回填/胜率。")
        conn.close()
        return

    print(f"已完成回填: {filled}")
    if has_hit:
        print(f"命中条数: {hit}")
        print(f"当前胜率: {winrate:.4f}")
    else:
        print("提示：predictions_1m 没有 hit_flag 列，仅显示回填数量。")
    print("==============================\n")

    print(f"最近{args.last}条（已完成回填）：\n")

    # 动态选择可用字段
    select_cols = ["ts"]
    if "symbol" in cols:
        select_cols.append("symbol")
    if "model" in cols:
        select_cols.append("model")
    for c in ["direction", "p_up", "score", "last_close", "future_close", "realized_ret", "hit_flag"]:
        if c in cols:
            select_cols.append(c)

    sel = ", ".join(select_cols)
    sql = f"""
        SELECT {sel}
        FROM predictions_1m
        {where_sql}
        {"AND future_close IS NOT NULL" if where_sql else "WHERE future_close IS NOT NULL"}
        ORDER BY ts DESC
        LIMIT ?
    """
    cur.execute(sql, params + [args.last])
    rows = cur.fetchall()

    if not rows:
        print("暂无已回填数据，请再运行几分钟 main.py。")
    else:
        # 简单格式化输出（按列名）
        for r in rows:
            row = dict(zip(select_cols, r))
            ts = row.get("ts")
            sym = row.get("symbol", "")
            model = row.get("model", "")
            direction = row.get("direction", "")
            p_up = row.get("p_up", None)
            score = row.get("score", None)
            lc = row.get("last_close", None)
            fc = row.get("future_close", None)
            rr = row.get("realized_ret", None)
            hf = row.get("hit_flag", None)

            head = f"{ts}"
            if sym:
                head += f" | {sym}"
            if model:
                head += f" | {model}"
            head += f" | {direction}"

            tail = []
            if p_up is not None:
                tail.append(f"p_up={float(p_up):.4f}")
            if score is not None:
                tail.append(f"score={float(score):.6f}")
            if (lc is not None) and (fc is not None):
                tail.append(f"{float(lc):.2f}->{float(fc):.2f}")
            if rr is not None:
                tail.append(f"ret={float(rr):.4f}")
            if hf is not None:
                tail.append(f"hit={int(hf)}")

            print(head + " | " + " | ".join(tail))

    conn.close()


if __name__ == "__main__":
    main()