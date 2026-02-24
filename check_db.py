import sqlite3

conn = sqlite3.connect("data/market.db")
cur = conn.cursor()

print("=== schemeA_simple_v1 分布 ===")
print(cur.execute("""
select strength, side, count(*)
from signals_1m
where model='schemeA_simple_v1'
group by strength, side
order by strength, side
""").fetchall())

print("\n=== baseline 分布 ===")
print(cur.execute("""
select strength, side, count(*)
from signals_1m
where model='baseline'
group by strength, side
order by strength, side
""").fetchall())

conn.close()