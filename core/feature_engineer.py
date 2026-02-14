import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ts 排序（无论 ts 是字符串还是时间戳都尽量处理）
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts")

    # 强制把价格列转成数值（关键修复点）
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 把关键列为空的行丢掉
    df = df.dropna(subset=["close"])

    # === 收益率 ===
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # === 均线偏离 ===
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_diff"] = df["close"] - df["ma_10"]

    # === 波动率 ===
    df["volatility_5"] = df["ret_1"].rolling(5).std()

    return df
