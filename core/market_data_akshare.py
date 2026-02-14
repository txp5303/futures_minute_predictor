from dataclasses import dataclass
from datetime import datetime
import akshare as ak


@dataclass
class Tick:
    symbol: str
    ts: datetime
    last: float
    volume: float | None = None
    open_interest: float | None = None


class AKShareMarketDataSource:
    def get_last_tick(self, symbol: str) -> Tick:
        try:
            df = ak.futures_zh_realtime(symbol=symbol)
        except KeyError as e:
            raise KeyError(
                f"AKShare 不支持 symbol='{symbol}'（需来自 futures_symbol_mark 的映射表）。原始错误: {e}"
            )

        row = df.iloc[0].to_dict()

        # 最新价：你这次返回列里有 trade（最常见的实时成交价字段）
        last = None
        for k in ["trade", "最新价", "最新", "最新价格", "last", "price", "close"]:
            if k in row and row[k] not in (None, "", "--"):
                try:
                    last = float(row[k])
                    break
                except Exception:
                    pass
        if last is None:
            raise ValueError(f"无法识别最新价字段，返回列={list(df.columns)}")

        # 成交量：你这次返回列里有 volume
        vol = None
        for k in ["volume", "成交量", "vol", "Volume"]:
            if k in row and row[k] not in (None, "", "--"):
                try:
                    vol = float(row[k])
                except Exception:
                    vol = None
                break

        # 持仓量：你这次返回列里有 position
        oi = None
        for k in ["position", "持仓量", "oi", "open_interest", "Position"]:
            if k in row and row[k] not in (None, "", "--"):
                try:
                    oi = float(row[k])
                except Exception:
                    oi = None
                break

        # 时间：优先用 ticktime（通常是 HH:MM:SS），拼成今天的完整时间
        ts = datetime.now()
        if "ticktime" in row and row["ticktime"] not in (None, "", "--"):
            try:
                t = str(row["ticktime"]).strip()
                today = datetime.now().strftime("%Y-%m-%d")
                ts = datetime.strptime(f"{today} {t}", "%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = datetime.now()

        return Tick(symbol=symbol, ts=ts, last=last, volume=vol, open_interest=oi)
