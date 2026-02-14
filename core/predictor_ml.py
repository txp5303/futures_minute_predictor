import joblib
import pandas as pd
from core.feature_engineer import generate_features


class MLPredictor:
    def __init__(self, model_path: str = "models/logistic_model.pkl", threshold: float = 0.5):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.features = payload["features"]
        self.threshold = threshold

    def predict_from_kline(self, kline_df: pd.DataFrame):
        """
        输入：kline_1m dataframe（至少包含 close/open/high/low/ts 等）
        输出：(direction, p_up, score) 或 None
        """
        df = generate_features(kline_df)
        df = df.dropna(subset=self.features)

        if df.empty:
            return None  # 数据还不够形成完整特征

        X = df.iloc[-1][self.features].to_frame().T

        # 类别1（上涨）的概率
        p_up = float(self.model.predict_proba(X)[0, 1])
        direction = "UP" if p_up >= self.threshold else "DOWN"
        score = p_up  # 暂时用 score 字段存概率

        return direction, p_up, score
