"""고장 예측 모델 클래스 (ML: 로지스틱회귀·랜덤포레스트 / DL: Keras 심층 신경망)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import CATEGORICAL_FEATURES, FAILURE_TYPES, NUMERIC_FEATURES
from .dl_model import KerasFailureClassifier

DL_NAME = "Keras DNN (Deep Learning)"


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])


class FailurePredictor:
    """1) 고장 여부(이진) 모델 3종 비교  2) 고장 유형(다중 라벨) 모델."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.binary_models: dict[str, Pipeline] = {
            "LogisticRegression": Pipeline([
                ("prep", _preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]),
            "RandomForest": Pipeline([
                ("prep", _preprocessor()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, class_weight="balanced_subsample",
                    random_state=random_state, n_jobs=-1)),
            ]),
            DL_NAME: Pipeline([
                ("prep", _preprocessor()),
                ("clf", KerasFailureClassifier(random_state=random_state)),
            ]),
        }
        self.type_model = Pipeline([
            ("prep", _preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                random_state=random_state, n_jobs=-1)),
        ])
        self.best_name: str | None = None
        self.metrics: dict = {}

    # ---------- 학습 ----------
    def fit(self, X_train, y_train, yt_train) -> "FailurePredictor":
        for model in self.binary_models.values():
            model.fit(X_train, y_train)
        self.type_model.fit(X_train, yt_train)
        return self

    # ---------- 평가 ----------
    def evaluate(self, X_test, y_test, yt_test) -> dict:
        results = {}
        for name, model in self.binary_models.items():
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            results[name] = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
                "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            }
        # 고장은 놓치면 안 되므로 F1 기준으로 대표 모델 선택
        self.best_name = max(results, key=lambda n: results[n]["f1"])

        type_pred = self.type_model.predict(X_test)
        type_results = {}
        for i, t in enumerate(FAILURE_TYPES):
            type_results[t] = {
                "support": int(yt_test[t].sum()),
                "recall": recall_score(yt_test[t], type_pred[:, i], zero_division=0),
                "precision": precision_score(yt_test[t], type_pred[:, i], zero_division=0),
                "f1": f1_score(yt_test[t], type_pred[:, i], zero_division=0),
            }
        self.metrics = {"binary": results, "best_model": self.best_name, "failure_types": type_results}
        return self.metrics

    def dl_history(self) -> dict:
        """Keras 학습 곡선 (epoch별 loss / val_loss / auc / val_auc)."""
        return self.binary_models[DL_NAME].named_steps["clf"].history_

    def feature_importance(self) -> pd.Series:
        rf = self.binary_models["RandomForest"]
        names = rf.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        return pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)

    # ---------- 예측 ----------
    def predict_one(self, X_row: pd.DataFrame, model_name: str | None = None) -> dict:
        name = model_name or self.best_name or "RandomForest"
        proba = float(self.binary_models[name].predict_proba(X_row)[0, 1])
        # 다중 출력 RF: 출력별 [P(0), P(1)] 리스트
        type_probas = {}
        for t, p in zip(FAILURE_TYPES, self.type_model.predict_proba(X_row)):
            type_probas[t] = float(p[0, 1]) if p.shape[1] > 1 else 0.0
        return {
            "model": name,
            "failure_probability": proba,
            "failure_predicted": proba >= 0.5,
            "failure_type_probabilities": type_probas,
        }

    # ---------- 저장/로드 ----------
    def save(self, path: str | Path = "models/predictor.joblib") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path = "models/predictor.joblib") -> "FailurePredictor":
        return joblib.load(path)
