from sqlalchemy import text
from core.db import ENGINE

SYMBOL = "螺纹钢"
MODEL = "baseline_sma"
HORIZON = 1

sql = """
SELECT
  p.symbol,
  p.ts AS pred_ts,
  p.direction AS pred_dir,
  p.score AS pred_score,
  k0.close AS close0,
  k1.close AS close1
FROM predictions_1m p
JOIN kline_1m k0
  ON k0.symbol = p.symbol AND k0.ts = p.ts
JOIN kline_1m k1
  ON k1.symbol = p.symbol AND k1.ts = datetime(p.ts, '+1 minute')
WHERE p.symbol = :symbol
  AND p.model = :model
  AND p.horizon = :horizon
ORDER BY p.ts DESC
LIMIT 500
"""

with ENGINE.begin() as conn:
    rows = conn.execute(text(sql), dict(symbol=SYMBOL, model=MODEL, horizon=HORIZON)).fetchall()

if not rows:
    print("没有可回测的数据：请先让系统跑几分钟，产生 predictions_1m 和下一分钟 kline_1m。")
    raise SystemExit(0)

total = 0
correct = 0

for r in rows:
    d = dict(r._mapping)
    close0 = float(d["close0"])
    close1 = float(d["close1"])
    true_dir = "UP" if close1 >= close0 else "DOWN"
    pred_dir = d["pred_dir"]

    total += 1
    if pred_dir == true_dir:
        correct += 1

acc = correct / total if total else 0.0
print(f"Symbol={SYMBOL} model={MODEL} horizon={HORIZON}")
print(f"Samples={total} Correct={correct} Accuracy={acc:.3f}")

# 打印最近10条对齐结果，方便你肉眼检查
print("\n最近10条：pred_ts | pred -> true | close0 -> close1 | score")
for r in rows[:10]:
    d = dict(r._mapping)
    close0 = float(d["close0"])
    close1 = float(d["close1"])
    true_dir = "UP" if close1 >= close0 else "DOWN"
    print(f"{d['pred_ts']} | {d['pred_dir']} -> {true_dir} | {close0:.2f}->{close1:.2f} | {float(d['pred_score']):.4f}")
