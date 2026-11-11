"""TensorFlow/Keras 심층 신경망(DNN) 고장 분류 클래스.

sklearn 방식(fit / predict / predict_proba)으로 감싸서, 다른 ML 모델과 같은 Pipeline 안에서
전처리 → 학습 → 평가를 똑같이 할 수 있게 했다.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow 안내 로그 줄이기


class KerasFailureClassifier(BaseEstimator, ClassifierMixin):
    """Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid) 구조의 이진 분류기.

    - 고장 데이터가 3.4%뿐이라 class_weight로 고장 샘플에 가중치를 준다.
    - EarlyStopping으로 검증 손실이 더 줄지 않으면 학습을 멈추고 가장 좋았던 가중치로 되돌린다.
    """

    def __init__(self, hidden_units=(64, 32), dropout=0.3, learning_rate=1e-3,
                 epochs=200, batch_size=64, patience=15, class_weight_power=0.5,
                 random_state=42, verbose=0):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.class_weight_power = class_weight_power  # 1.0=완전 균형, 0.5=제곱근(오경보 억제)
        self.random_state = random_state
        self.verbose = verbose

    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers

        model = tf.keras.Sequential([layers.Input(shape=(n_features,))])
        for units in self.hidden_units:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])

        balanced = compute_class_weight("balanced", classes=self.classes_, y=y.astype(int))
        weights = balanced ** self.class_weight_power
        class_weight = {0: float(weights[0]), 1: float(weights[1])}

        self.model_ = self._build(X.shape[1])
        history = self.model_.fit(
            X, y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True)],
            verbose=self.verbose,
        )
        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ---------- joblib 저장 지원: Keras 모델은 .keras 파일 바이트로 바꿔서 저장 ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.keras")
                model.save(path)
                with open(path, "rb") as f:
                    state["_model_bytes"] = f.read()
        return state

    def __setstate__(self, state):
        data = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if data is not None:
            import tensorflow as tf
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.keras")
                with open(path, "wb") as f:
                    f.write(data)
                self.model_ = tf.keras.models.load_model(path)
