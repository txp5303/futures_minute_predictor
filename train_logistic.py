import sqlite3
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from core.feature_engineer import generate_features

DB_PATH = "data/market.db"
MODEL_PATH = "models/logistic_model.pkl"


def main():
    # === 读取数据库 ===
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM kline_1m", conn)
    conn.close()

    if df.empty:
        raise RuntimeError("kline_1m 表为空，请先运行 main.py 生成数据。")

    # === 生成特征 ===
    df = generate_features(df)

    # === 构造标签（下一分钟是否上涨）===
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # === 特征列（先定义，后面 dropna 只针对这些列）===
    features = ["ret_1", "ret_5", "ret_10", "ma_diff", "volatility_5"]

    # ✅ 关键修复：不要 df.dropna()，只清理训练需要的列
    df = df.dropna(subset=features + ["target"])

    print("有效样本数:", len(df))
    print("NaN统计（应全为0）：")
    print(df[features + ["target"]].isna().sum())

    # 小样本也允许跑通流程（你目前只有 31 根K线）
    if len(df) < 10:
        raise RuntimeError(f"有效样本太少（{len(df)}行）：请先积累更多K线或减少rolling窗口。")

    X = df[features]
    y = df["target"]

    # === 时间序列切分（保证测试集至少5条）===
    split = int(len(df) * 0.8)
    if len(df) - split < 5:
        split = max(len(df) - 5, 1)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # === 训练模型 ===
    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_train, y_train)

    # === 测试 ===
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Train={len(X_train)} Test={len(X_test)}")
    print("Accuracy:", round(acc, 4))

    # === 保存模型（含特征名，便于实时推理对齐）===
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print("模型已保存:", MODEL_PATH)


if __name__ == "__main__":
    main()
