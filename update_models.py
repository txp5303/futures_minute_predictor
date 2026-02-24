import sqlite3
from pathlib import Path

db_path = Path("data/market.db")
if not db_path.exists():
    print("数据库不存在：", db_path)
    raise SystemExit(1)

con = sqlite3.connect(db_path)
cur = con.cursor()

# 把 schemeA_simple_v1 统一改名为 schemeA+
cur.execute("UPDATE signals_1m SET model=? WHERE model=?", ("schemeA+", "schemeA_simple_v1"))
con.commit()

print("updated rows:", cur.rowcount)

con.close()