from dataclasses import dataclass
from datetime import datetime
import random


@dataclass
class Tick:
    symbol: str
    ts: datetime
    last: float
    volume: float | None = None
    open_interest: float | None = None


class DemoMarketDataSource:
    """
    休市/调试用：模拟跳动价格
    """
    def __init__(self, base: float = 3055.0):
        self.base = base

    def get_last_tick(self, symbol: str) -> Tick:
        now = datetime.now()
        # 小幅随机波动
        self.base = self.base + random.uniform(-2, 2)
        last = float(self.base)
        return Tick(symbol=symbol, ts=now, last=last, volume=None, open_interest=None)
