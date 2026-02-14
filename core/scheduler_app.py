import logging
import sqlite3
import pandas as pd
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from core.predictor_ml import MLPredictor


DB_PATH = "data/market.db"


# =========================
# 工具函数
# =========================

def load_recent_kline(symbol: str, limit: int = 200) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM kline_1m WHERE symbol=? ORDER BY ts ASC",
        conn,
        params=(symbol,),
    )
    conn.close()
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df


def insert_prediction(ts: str, horizon: int, direction: str, score: float, model: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions_1m (ts, horizon, direction, score, model)
        VALUES (?, ?, ?, ?, ?)
    """, (ts, horizon, direction, score, model))
    conn.commit()
    conn.close()


# =========================
# ML预测器初始化
# =========================

ml_predictor = None


def init_ml():
    global ml_predictor
    ml_predictor = MLPredictor(
        model_path="models/logistic_model.pkl",
        threshold=0.5
    )
    logging.info("ML Predictor 已加载 ✅")


# =========================
# 每分钟执行的任务
# =========================

def minute_task(keyword: str):
    """
    这里假设：
    你的系统已经在别处完成：
    1) tick抓取
    2) 聚合成1分钟K线
    3) 写入 kline_1m
    """

    symbol = "RB9999"  # 根据你现在 demo 逻辑写死

    kdf = load_recent_kline(symbol)

    if kdf.empty:
        return

    last_ts = kdf.iloc[-1]["ts"]

    res = ml_predictor.predict_from_kline(kdf)

    if res:
        direction, p_up, score = res

        insert_prediction(
            ts=last_ts,
            horizon=1,
            direction=direction,
            score=score,
            model="logistic_v1"
        )

        logging.info(
            "[ML] %s %s dir=%s p_up=%.4f",
            last_ts,
            symbol,
            direction,
            p_up
        )


# =========================
# 启动调度器
# =========================

def run_scheduler(keyword: str = "螺纹"):
    init_ml()

    scheduler = BlockingScheduler()

    # 每60秒执行一次
    scheduler.add_job(
        minute_task,
        "interval",
        seconds=60,
        args=[keyword],
        next_run_time=datetime.now()
    )

    logging.info("调度器启动，每分钟执行一次预测任务...")
    scheduler.start()
