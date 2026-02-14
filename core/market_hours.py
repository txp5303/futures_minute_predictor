from datetime import datetime

def is_market_open_cn_futures(now: datetime | None = None) -> bool:
    """
    简化版：周一~周五算可抓，周末休市直接跳过。
    """
    now = now or datetime.now()
    return now.weekday() <= 4
