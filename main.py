from datetime import datetime
from pathlib import Path
import logging

BASE_DIR = Path(__file__).resolve().parent


def setup_dirs():
    for d in ["data", "logs", "models", "strategies", "core"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)


def setup_logger():
    log_file = BASE_DIR / "logs" / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logging.info("Logger ready. Log file: %s", log_file)


def main():
    setup_dirs()
    setup_logger()

    # 降低第三方库(apscheduler)日志噪音
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    # 1) 初始化数据库
    from core.db import init_db
    init_db()
    logging.info("数据库初始化完成 ✅ data/market.db")

    # 2) 基本启动信息
    logging.info("系统启动成功 ✅")
    logging.info("启动时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("项目目录: %s", BASE_DIR)

    # 3) 进入 AKShare 实时行情定时采集模式
    # keyword 可以改成：螺纹 / 热卷 / 铁矿 / 豆粕 / 沪铜 等
    from core.scheduler_app import run_scheduler
    run_scheduler(keyword="螺纹")


if __name__ == "__main__":
    main()
