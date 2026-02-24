import sqlite3
from pathlib import Path


# 数据库路径（自动定位到 data/market.db）
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "market.db"


def main():
    if not DB_PATH.exists():
        print("数据库不存在，请先运行 main.py 生成数据。")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 总预测数
    cur.execute("SELECT COUNT(*) FROM predictions_1m")
    total = cur.fetchone()[0]

    # 已回填数量
    cur.execute("SELECT COUNT(*) FROM predictions_1m WHERE future_close IS NOT NULL")
    filled = cur.fetchone()[0]

    # 命中数量
    cur.execute("SELECT COUNT(*) FROM predictions_1m WHERE hit_flag = 1")
    hit = cur.fetchone()[0]

    winrate = (hit / filled) if filled else 0.0

    print("\n==============================")
    print("   在线预测统计结果")
    print("==============================")
    print(f"数据库路径: {DB_PATH}")
    print(f"总预测条数: {total}")
    print(f"已完成回填: {filled}")
    print(f"命中条数: {hit}")
    print(f"当前胜率: {winrate:.4f}")
    print("==============================\n")

    # 最近10条已回填记录
    print("最近10条（已完成回填）：\n")

    cur.execute("""
        SELECT ts, direction, p_up, score, last_close, future_close, realized_ret, hit_flag
        FROM predictions_1m
        WHERE future_close IS NOT NULL
        ORDER BY ts DESC
        LIMIT 10
    """)

    rows = cur.fetchall()

    if not rows:
        print("暂无已回填数据，请再运行几分钟 main.py。")
    else:
        for r in rows:
            ts, direction, p_up, score, last_close, future_close, realized_ret, hit_flag = r
            print(
                f"{ts} | {direction} | "
                f"p_up={p_up:.4f} | score={score:.6f} | "
                f"{last_close:.2f}->{future_close:.2f} | "
                f"ret={realized_ret:.4f} | hit={hit_flag}"
            )

    conn.close()


if __name__ == "__main__":
    main()