"""AI4I 2020 데이터 로드 · 전처리 · 특성공학 클래스."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}

BASE_FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
ENGINEERED_FEATURES = ["temp_diff", "power", "strain"]
NUMERIC_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["type"]
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "machine_failure"


class AI4IDataLoader:
    """CSV를 읽고, 컬럼명을 정리하고, 도메인 특성을 추가한 뒤 학습/테스트로 나눈다."""

    def __init__(self, csv_path: str | Path = "data/ai4i2020.csv", random_state: int = 42):
        self.csv_path = Path(csv_path)
        self.random_state = random_state
        self.df: pd.DataFrame | None = None

    # ---------- 로드 ----------
    def load(self) -> pd.DataFrame:
        if self.csv_path.exists():
            raw = pd.read_csv(self.csv_path)
        else:  # 로컬 파일이 없으면 UCI에서 직접 받는다 (Colab/Streamlit Cloud 대비)
            raw = pd.read_csv(DATA_URL, compression="zip")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(self.csv_path, index=False)
        df = raw.rename(columns=RENAME_MAP).drop(columns=["UDI", "Product ID"])
        self.df = self.add_features(df)
        return self.df

    # ---------- 특성공학 ----------
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """고장 메커니즘에 맞춘 파생 변수.

        - temp_diff : 공정온도 - 공기온도 (열 방출 고장 HDF 관련)
        - power     : 토크 × 각속도 [W] (전력 고장 PWF 관련)
        - strain    : 공구마모 × 토크 [min·Nm] (과부하 고장 OSF 관련)
        """
        df = df.copy()
        df["temp_diff"] = df["process_temp"] - df["air_temp"]
        df["power"] = df["torque"] * df["rpm"] * 2 * math.pi / 60
        df["strain"] = df["tool_wear"] * df["torque"]
        return df

    @classmethod
    def make_single(cls, type_: str, air_temp: float, process_temp: float,
                    rpm: float, torque: float, tool_wear: float) -> pd.DataFrame:
        """UI 입력값 한 건을 모델 입력 형태(DataFrame)로 변환."""
        row = pd.DataFrame([{
            "type": type_, "air_temp": air_temp, "process_temp": process_temp,
            "rpm": rpm, "torque": torque, "tool_wear": tool_wear,
        }])
        return cls.add_features(row)

    # ---------- 분할 ----------
    def split(self, test_size: float = 0.2):
        if self.df is None:
            self.load()
        X = self.df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = self.df[TARGET]
        y_types = self.df[FAILURE_TYPES]
        return train_test_split(
            X, y, y_types, test_size=test_size, stratify=y, random_state=self.random_state
        )  # X_train, X_test, y_train, y_test, yt_train, yt_test

    def summary(self) -> dict:
        if self.df is None:
            self.load()
        return {
            "rows": len(self.df),
            "failure_rate": float(self.df[TARGET].mean()),
            "failure_type_counts": {t: int(self.df[t].sum()) for t in FAILURE_TYPES},
        }
