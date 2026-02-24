import sqlite3
from pathlib import Path

db_path = Path("data/market.db")

if not db_path.exists():
    print("数据库不存在：", db_path)
    exit()

con = sqlite3.connect(db_path)
cur = con.cursor()

rows = cur.execute("""
SELECT IFNULL(model, 'NULL') AS model_name, COUNT(*)
FROM signals_1m
GROUP BY IFNULL(model, 'NULL')
ORDER BY COUNT(*) DESC
""").fetchall()

print("signals_1m 中的 model 分布：")
print(rows)

con.close()