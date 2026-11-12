"""Keras DNN 실험 클래스: 전처리(파생 변수)와 모델 설정을 하나씩 바꿔 성능을 비교한다.

기준 설정에서 한 가지만 바꾼 실험을 같은 데이터 분할로 학습해, 각 선택이 성능에 준 영향을 확인한다.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import BASE_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .dl_model import KerasFailureClassifier

BASELINE = {"features": "base+derived", "hidden_units": (64, 32), "dropout": 0.3, "class_weight_power": 0.5}

# (실험 이름, 실험 그룹, 기준 설정에서 바꾼 값)
EXPERIMENTS = [
    ("기준 모델 (64-32)", "기준", {}),
    ("파생 변수 제외", "전처리", {"features": "base"}),
    ("은닉층 1개 (32)", "구조", {"hidden_units": (32,)}),
    ("은닉층 3개 (128-64-32)", "구조", {"hidden_units": (128, 64, 32)}),
    ("Dropout 없음", "규제", {"dropout": 0.0}),
    ("클래스 가중치 없음", "불균형", {"class_weight_power": 0.0}),
    ("클래스 가중치 완전 균형", "불균형", {"class_weight_power": 1.0}),
]


class DLExperiment:
    """EXPERIMENTS의 각 설정으로 Keras DNN을 학습 · 평가한다."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results: list[dict] = []

    @staticmethod
    def _pipeline(cfg: dict, random_state: int) -> Pipeline:
        numeric = NUMERIC_FEATURES if cfg["features"] == "base+derived" else BASE_FEATURES
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), numeric),
        ])
        clf = KerasFailureClassifier(hidden_units=cfg["hidden_units"], dropout=cfg["dropout"],
                                     class_weight_power=cfg["class_weight_power"], random_state=random_state)
        return Pipeline([("prep", prep), ("clf", clf)])

    def run(self, X_train, X_test, y_train, y_test, log=print) -> list[dict]:
        self.results = []
        for name, group, change in EXPERIMENTS:
            cfg = {**BASELINE, **change}
            model = self._pipeline(cfg, self.random_state).fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            res = {
                "name": name, "group": group,
                "f1": f1_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
                "epochs": len(model.named_steps["clf"].history_["loss"]),
                "false_alarms": int(((pred == 1) & (y_test == 0)).sum()),
                "missed": int(((pred == 0) & (y_test == 1)).sum()),
            }
            self.results.append(res)
            log(f"{name:<22} F1 {res['f1']:.3f}  재현율 {res['recall']:.3f}  정밀도 {res['precision']:.3f}  "
                f"AUC {res['roc_auc']:.3f}  epoch {res['epochs']}")
        return self.results
