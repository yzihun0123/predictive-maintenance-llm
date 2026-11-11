"""모델 학습 · 평가 · 결과 저장 스크립트.

실행: python train.py
결과: models/predictor.joblib, outputs/metrics.json, outputs/*.png (보고서용 그림)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import AI4IDataLoader
from src.models import FailurePredictor

OUT = Path("outputs")


def main():
    loader = AI4IDataLoader()
    loader.load()
    print("데이터 요약:", loader.summary())

    X_train, X_test, y_train, y_test, yt_train, yt_test = loader.split()
    predictor = FailurePredictor().fit(X_train, y_train, yt_train)
    metrics = predictor.evaluate(X_test, y_test, yt_test)
    predictor.save()

    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[고장 여부 모델 비교]")
    for name, m in metrics["binary"].items():
        print(f"{name:22s} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    print("대표 모델:", metrics["best_model"])
    print("\n[고장 유형별 (RandomForest)]")
    for t, m in metrics["failure_types"].items():
        print(f"{t}: support={m['support']} rec={m['recall']:.3f} f1={m['f1']:.3f}")

    # 그림 1: 모델 비교
    names = list(metrics["binary"])
    keys = ["precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.2
    for i, k in enumerate(keys):
        ax.bar([x + i * w for x in range(len(names))], [metrics["binary"][n][k] for n in names], w, label=k)
    ax.set_xticks([x + 1.5 * w for x in range(len(names))], names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model comparison (test set)")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150)

    # 그림 2: 특성 중요도
    fi = predictor.feature_importance()
    fig, ax = plt.subplots(figsize=(7, 4))
    fi.iloc[::-1].plot.barh(ax=ax)
    ax.set_title("RandomForest feature importance")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)

    # 그림 3: Keras 학습 곡선
    hist = predictor.dl_history()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["loss"], label="train"); axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set_title("Keras DNN loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(hist["auc"], label="train"); axes[1].plot(hist["val_auc"], label="validation")
    axes[1].set_title("Keras DNN AUC"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "dl_learning_curve.png", dpi=150)
    print(f"Keras 학습 epoch 수: {len(hist['loss'])}")
    print("\n저장 완료: models/predictor.joblib, outputs/")


if __name__ == "__main__":
    main()
