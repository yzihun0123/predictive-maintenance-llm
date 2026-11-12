"""딥러닝 실험: 파생 변수 · 은닉층 구조 · Dropout · 클래스 가중치를 하나씩 바꿔 비교.

실행: python dl_experiments.py              (CPU 기준 수 분)
      python dl_experiments.py --plot-only  (저장된 결과로 그래프만 다시 그림)
결과: outputs/dl_experiments.json, outputs/dl_experiments.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RESULT = OUT / "dl_experiments.json"

if "--plot-only" in sys.argv and RESULT.exists():
    results = json.loads(RESULT.read_text(encoding="utf-8"))
else:
    from src.data_loader import AI4IDataLoader
    from src.dl_experiment import DLExperiment

    X_train, X_test, y_train, y_test, _, _ = AI4IDataLoader().split()
    results = DLExperiment().run(X_train, X_test, y_train, y_test)
    RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
ORANGE, CHAR, GRAY = "#E8590C", "#2B3138", "#9AA0A6"
r = results[::-1]  # 기준 모델을 맨 위에
names = [x["name"] for x in r]
y = np.arange(len(r))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
f1 = [x["f1"] for x in r]
bars = ax1.barh(y, f1, color=[ORANGE if x["group"] == "기준" else GRAY for x in r])
for b, v in zip(bars, f1):
    ax1.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=11)
ax1.set_yticks(y, names, fontsize=11)
ax1.set_xlim(0, 0.85)
ax1.set_title("F1 (높을수록 좋음)", fontsize=13, fontweight="bold")

h = 0.38
missed = [x["missed"] for x in r]
alarms = [x["false_alarms"] for x in r]
b1 = ax2.barh(y + h / 2, missed, h, color=CHAR, label="놓친 고장")
b2 = ax2.barh(y - h / 2, alarms, h, color=ORANGE, label="잘못된 경보")
for bs, vals in ((b1, missed), (b2, alarms)):
    for b, v in zip(bs, vals):
        ax2.text(v + 1.5, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=10)
ax2.set_yticks(y, [""] * len(r))
ax2.set_xlim(0, max(alarms) * 1.15)
ax2.set_title("테스트 2,000건 중 오류 건수 (고장 68건)", fontsize=13, fontweight="bold")
ax2.legend(frameon=False, loc="upper right")
for ax in (ax1, ax2):
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "dl_experiments.png", dpi=170)
print("saved: outputs/dl_experiments.json, outputs/dl_experiments.png")
