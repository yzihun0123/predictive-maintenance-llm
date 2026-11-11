"""AI4I 데이터셋 논문(Matzka, 2020)에 정의된 고장 조건으로 위험 신호를 점검하는 클래스.

ML 모델의 예측 근거를 LLM에게 함께 넘겨서, 리포트가 '왜' 위험한지 설명할 수 있게 한다.
"""
from __future__ import annotations

OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}  # 제품 등급별 과부하 한계 [min·Nm]

FAILURE_NAMES_KO = {
    "TWF": "공구 마모",
    "HDF": "열 방출 불량",
    "PWF": "전력 이상",
    "OSF": "과부하",
    "RNF": "원인 불명",
}

FAILURE_DESC_KO = {
    "TWF": "공구를 오래 써서 날이 닳아 생기는 고장",
    "HDF": "가공 열이 잘 빠지지 않아 생기는 고장 (온도차가 작고 회전이 느릴 때)",
    "PWF": "모터에 걸리는 전력이 너무 적거나 많을 때 생기는 고장",
    "OSF": "닳은 공구로 세게 깎을 때 생기는 고장",
    "RNF": "특별한 원인 없이 낮은 확률로 생기는 고장",
}


class FailureRuleChecker:
    """센서값 한 건에 대해 고장 유형별 기준 충족 여부를 계산."""

    def check(self, row: dict) -> list[dict]:
        t = row["type"]
        findings = [
            {
                "code": "TWF",
                "triggered": 200 <= row["tool_wear"] <= 240,
                "value": f"{row['tool_wear']:.0f}분 사용",
                "criterion": "200~240분",
            },
            {
                "code": "HDF",
                "triggered": row["temp_diff"] < 8.6 and row["rpm"] < 1380,
                "value": f"{row['temp_diff']:.1f}°C / {row['rpm']:.0f}rpm",
                "criterion": "8.6°C 미만 + 1380rpm 미만",
            },
            {
                "code": "PWF",
                "triggered": row["power"] < 3500 or row["power"] > 9000,
                "value": f"{row['power']:,.0f}W",
                "criterion": "3,500W 미만 / 9,000W 초과",
            },
            {
                "code": "OSF",
                "triggered": row["strain"] > OSF_LIMIT[t],
                "value": f"부하 {row['strain']:,.0f}",
                "criterion": f"{t}등급 {OSF_LIMIT[t]:,} 초과",
            },
        ]
        for f in findings:
            f["name"] = FAILURE_NAMES_KO[f["code"]]
            f["detail"] = f"{f['value']} (위험 기준: {f['criterion']})"
        return findings
