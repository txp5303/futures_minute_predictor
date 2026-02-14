from dataclasses import dataclass
from typing import Optional


@dataclass
class Bar1m:
    symbol: str
    ts_min: str  # YYYY-mm-dd HH:MM:00
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0              # 本分钟成交量增量
    open_interest: Optional[float] = None  # 本分钟最后持仓
    source: str = "agg_tick"


class MinuteAggregator:
    """
    聚合 tick -> 1分钟K线（OHLC + volume增量 + 最后持仓）
    update(ts_min, price, volume, open_interest)
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_min: Optional[str] = None

        self.o = self.h = self.l = self.c = None

        # 成交量：记录分钟内首/末，用于计算增量
        self.vol_first: Optional[float] = None
        self.vol_last: Optional[float] = None

        # 持仓量：取最后一次
        self.oi_last: Optional[float] = None

    def update(self, ts_min: str, price: float, volume: Optional[float] = None, open_interest: Optional[float] = None):
        if self.current_min is None:
            self.current_min = ts_min
            self.o = self.h = self.l = self.c = price

            self.vol_first = volume if volume is not None else None
            self.vol_last = volume if volume is not None else None
            self.oi_last = open_interest
            return

        if ts_min == self.current_min:
            self.c = price
            if price > self.h:
                self.h = price
            if price < self.l:
                self.l = price

            if volume is not None:
                if self.vol_first is None:
                    self.vol_first = volume
                self.vol_last = volume

            if open_interest is not None:
                self.oi_last = open_interest
            return

        raise RuntimeError(f"Minute changed: {self.current_min} -> {ts_min}")

    def flush(self) -> Optional[Bar1m]:
        if self.current_min is None:
            return None

        vol_delta = 0.0
        if self.vol_first is not None and self.vol_last is not None:
            try:
                vol_delta = float(self.vol_last) - float(self.vol_first)
                if vol_delta < 0:
                    vol_delta = 0.0
            except Exception:
                vol_delta = 0.0

        bar = Bar1m(
            symbol=self.symbol,
            ts_min=self.current_min,
            open=float(self.o),
            high=float(self.h),
            low=float(self.l),
            close=float(self.c),
            volume=vol_delta,
            open_interest=self.oi_last,
            source="agg_tick",
        )

        # 清空
        self.current_min = None
        self.o = self.h = self.l = self.c = None
        self.vol_first = None
        self.vol_last = None
        self.oi_last = None
        return bar
