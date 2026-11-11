"""LLM 정비 상담 에이전트.

작업자의 자연어 질문을 LLM이 이해하고, 필요한 도구(ML/DL 고장 예측, 조정안 탐색)를 스스로 호출한 뒤
도구가 계산한 수치만 근거로 답한다.

    질문 → LLM(bind_tools) → 도구 호출 → 결과(ToolMessage) → LLM → … → 최종 답변

설계 원칙
- 확률·판정·조정 효과는 반드시 도구(학습된 모델)가 계산한다. LLM은 숫자를 만들지 않는다.
- 답변에 도구 결과에 없는 퍼센트 수치가 나오면 check_numbers()가 표시한다 (환각 감지).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from .data_loader import AI4IDataLoader
from .llm_reporter import MaintenanceReporter
from .rule_checker import FAILURE_DESC_KO, FAILURE_NAMES_KO, OSF_LIMIT, FailureRuleChecker

K = 273.15
DEFAULTS = {"air_temp_c": 26.9, "process_temp_c": 36.9, "rpm": 1500}  # 학습 데이터 중앙값 근처
RANGES = {  # 학습 데이터 범위
    "rpm": (1150, 2900), "torque": (3, 77), "tool_wear_min": (0, 255),
    "air_temp_c": (21.9, 31.9), "process_temp_c": (31.9, 40.9),
}

SYSTEM_PROMPT = f"""너는 공장의 밀링 머신(금속을 깎는 기계) 정비 상담 담당자다. 현장 작업자의 질문에 답한다.

[도구 사용 규칙]
- 설비가 괜찮은지, 고장 확률, 값을 바꾸면 어떻게 되는지는 반드시 도구로 계산한다. 직접 추측하지 않는다.
- diagnose: 현재 상태 진단. 값을 바꿔 보는 질문("토크를 45로 낮추면?")도 바꾼 값으로 diagnose를 다시 호출한다.
- suggest_adjustments: "어떻게 하면 돼?", "뭘 바꿔야 해?"처럼 해결책을 물을 때 호출한다.
- 제품 등급, 토크, 공구 사용 시간은 필수다. 하나라도 모르면 도구를 부르지 말고 먼저 물어본다.
- 회전수와 온도는 모르면 비워서 호출한다. 도구가 평균값을 가정한다.
- 도구 결과의 assumed_defaults가 "없음"이 아니면, 답변 마지막 줄에 "가정: (그 내용)"을 반드시 쓴다.
  가정한 값 때문에 생긴 위험 요인(예: 회전수를 가정해서 계산된 전력)은 그 사실을 함께 밝힌다.
- 이전 대화에서 나온 값은 다시 묻지 말고 이어서 쓴다.

[답변 규칙]
- 숫자(확률, 판정)는 도구 결과에 나온 것만 쓴다. 판정은 도구의 verdict를 그대로 쓴다.
- 첫 줄에 결론(판정과 고장 확률), 그 다음 근거, 마지막에 할 일. 짧은 개조식.
- 개조식(~함, ~필요, ~권장)으로 쓴다. "~습니다", "~해 보시기 바랍니다" 같은 말투, 인사말, 이모지 금지. 6~10줄 이내.
- 학습 범위 밖 값이면 예측을 믿기 어렵다고 알린다.
- 밀링 머신 정비와 무관한 질문은 범위 밖이라고 짧게 답한다.

[배경 지식]
- 제품 등급 L/M/H: 품질 기준. 등급이 높을수록 견딜 수 있는 공구 부하가 크다.
- 고장 유형: """ + "; ".join(f"{FAILURE_NAMES_KO[k]}={FAILURE_DESC_KO[k]}" for k in FAILURE_NAMES_KO) + f"""
- 판정 기준: 고장 확률 50% 이상 위험, 20% 이상이거나 기준 초과 항목이 있으면 주의, 나머지 정상.
- 입력 범위: 회전수 1150~2900rpm, 토크 3~77Nm, 공구 사용 0~255분, 주변 21.9~31.9°C, 가공 31.9~40.9°C.
- 과부하 한계(공구 사용 시간×토크): L {OSF_LIMIT['L']:,}, M {OSF_LIMIT['M']:,}, H {OSF_LIMIT['H']:,}.
"""


class MaintenanceAgent:
    """학습된 FailurePredictor를 도구로 쓰는 LLM 에이전트."""

    def __init__(self, predictor, provider: str = "openai", api_key: str | None = None,
                 model: str | None = None, temperature: float = 0.1, max_steps: int = 6,
                 ollama_url: str = "http://localhost:11434"):
        self.predictor = predictor
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or ("gpt-4o-mini" if provider == "openai" else "gemma4:e2b")
        self.temperature = temperature
        self.max_steps = max_steps
        self.ollama_url = ollama_url
        self.rules = FailureRuleChecker()
        self.tools = self._make_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self._llm = None

    # ------------------------------------------------------------------ 도구
    def _row(self, product_type, torque, tool_wear_min, rpm, air_temp_c, process_temp_c):
        return AI4IDataLoader.make_single(product_type, air_temp_c + K, process_temp_c + K,
                                          rpm, torque, tool_wear_min)

    def _fill(self, args: dict) -> tuple[dict, list[str], list[str]]:
        """빠진 값은 평균값으로 채우고, 학습 범위를 벗어난 값을 찾는다."""
        assumed, out_of_range = [], []
        grade = str(args.get("product_type", "")).strip().upper()[:1]
        if grade not in ("L", "M", "H"):
            raise ValueError("제품 등급은 L, M, H 중 하나여야 함. 작업자에게 등급을 물어볼 것")
        for k, v in DEFAULTS.items():
            if args.get(k) is None:
                args[k] = v
                assumed.append(k)
        for k, (lo, hi) in RANGES.items():
            if not lo <= args[k] <= hi:
                out_of_range.append(f"{k}={args[k]} (학습 범위 {lo}~{hi})")
        args["product_type"] = grade
        labels = {"air_temp_c": ("주변 온도", "°C"), "process_temp_c": ("가공 온도", "°C"),
                  "rpm": ("회전수", "rpm")}
        assumed = [f"{labels[k][0]} {args[k]:g}{labels[k][1]}" for k in assumed]
        return args, assumed, out_of_range

    @staticmethod
    def _assumed_text(assumed: list[str]) -> str:
        return (", ".join(assumed) + " 으로 가정함 (실제 값을 알려주면 다시 계산)") if assumed else "없음"

    def diagnose_values(self, **args) -> dict:
        args, assumed, out_of_range = self._fill(dict(args))
        X = self._row(args["product_type"], args["torque"], args["tool_wear_min"],
                      args["rpm"], args["air_temp_c"], args["process_temp_c"])
        sensor = X.iloc[0].to_dict()
        pred = self.predictor.predict_one(X)
        findings = self.rules.check(sensor)
        return {
            "verdict": MaintenanceReporter.judge(pred, findings),
            "failure_probability": f"{pred['failure_probability']:.1%}",
            "over_limit": [{"유형": f["name"], "현재값": f["value"], "위험기준": f["criterion"]}
                           for f in findings if f["triggered"]] or "없음",
            "type_probability": {FAILURE_NAMES_KO[k]: f"{v:.1%}"
                                 for k, v in pred["failure_type_probabilities"].items() if v >= 0.05},
            "derived": {"온도차_C": round(sensor["temp_diff"], 1), "소비전력_W": round(sensor["power"]),
                        "공구부하": round(sensor["strain"])},
            "inputs": args,
            "assumed_defaults": self._assumed_text(assumed),
            "out_of_training_range": out_of_range or "없음",
            "model": pred["model"],
        }

    def _batch_safe(self, base: dict, key: str, values) -> list[tuple[float, float, bool]]:
        """key 값을 바꿔가며 한 번에 예측. (값, 고장확률, 기준초과여부) 목록 반환."""
        rows = []
        for v in values:
            a = {**base, key: v}
            rows.append(self._row(a["product_type"], a["torque"], a["tool_wear_min"],
                                  a["rpm"], a["air_temp_c"], a["process_temp_c"]))
        X = pd.concat(rows, ignore_index=True)
        model = self.predictor.binary_models[self.predictor.best_name or "RandomForest"]
        probs = model.predict_proba(X)[:, 1]
        out = []
        for v, p, (_, r) in zip(values, probs, X.iterrows()):
            # 공구 마모 구간(TWF)은 토크·회전수로는 풀 수 없고 공구 교체로만 해소되므로 여기서는 제외
            over = any(f["triggered"] for f in self.rules.check(r.to_dict()) if f["code"] != "TWF")
            out.append((float(v), float(p), over))
        return out

    def suggest_values(self, **args) -> dict:
        args, assumed, _ = self._fill(dict(args))
        current = self.diagnose_values(**args)
        if current["verdict"] == "정상":
            return {"current": {"verdict": "정상", "failure_probability": current["failure_probability"]},
                    "options": "조정 불필요 (현재 정상)",
                    "assumed_defaults": self._assumed_text(assumed)}
        options = []

        # 1) 공구 교체
        tool = self.diagnose_values(**{**args, "tool_wear_min": 0})
        options.append({"조치": "공구 교체 (사용 시간 0분)", "판정": tool["verdict"],
                        "고장확률": tool["failure_probability"]})

        # 2) 토크 낮추기: 현재보다 낮은 값 중 안전(20% 미만, 기준 초과 없음)한 가장 높은 토크
        torques = np.arange(args["torque"] - 2.5, 2.9, -2.5)
        if len(torques):
            safe = [(v, p) for v, p, over in self._batch_safe(args, "torque", torques)
                    if p < 0.2 and not over]
            if safe:
                v, _ = safe[0]
                d = self.diagnose_values(**{**args, "torque": float(v)})
                options.append({"조치": f"토크 {args['torque']:.1f} → {v:.1f}Nm", "판정": d["verdict"],
                                "고장확률": d["failure_probability"]})
            else:
                options.append({"조치": "토크 낮추기", "결과": "토크만 낮춰서는 안전 구간을 찾지 못함"})

        # 3) 회전수 조정: 현재 값과 가장 가까운 안전한 회전수
        rpms = np.arange(1150, 2901, 50)
        safe = [(v, p) for v, p, over in self._batch_safe(args, "rpm", rpms) if p < 0.2 and not over]
        if safe:
            v, _ = min(safe, key=lambda t: abs(t[0] - args["rpm"]))
            d = self.diagnose_values(**{**args, "rpm": float(v)})
            options.append({"조치": f"회전수 {args['rpm']:.0f} → {v:.0f}rpm", "판정": d["verdict"],
                            "고장확률": d["failure_probability"]})
        else:
            options.append({"조치": "회전수 조정", "결과": "회전수만 바꿔서는 안전 구간을 찾지 못함"})

        notes = []
        if 200 <= args["tool_wear_min"] <= 240:
            notes.append("공구 사용 시간이 200~240분 구간이라 공구 마모 위험은 공구 교체로만 해소됨")
        return {"current": {"verdict": current["verdict"],
                            "failure_probability": current["failure_probability"]},
                "options": options, "notes": notes or "없음",
                "assumed_defaults": self._assumed_text(assumed)}

    def _make_tools(self):
        from langchain_core.tools import tool

        agent = self

        @tool
        def diagnose(product_type: str, torque: float, tool_wear_min: float,
                     rpm: float | None = None, air_temp_c: float | None = None,
                     process_temp_c: float | None = None) -> str:
            """밀링 머신 상태를 학습된 모델로 진단한다. 판정(정상/주의/위험), 고장 확률, 기준 초과 항목을 돌려준다.

            Args:
                product_type: 제품 등급 L, M, H 중 하나
                torque: 토크 (Nm)
                tool_wear_min: 공구 사용 시간 (분)
                rpm: 회전수 (rpm), 모르면 비움
                air_temp_c: 주변 온도 (섭씨), 모르면 비움
                process_temp_c: 가공 온도 (섭씨), 모르면 비움
            """
            return json.dumps(agent.diagnose_values(
                product_type=product_type, torque=torque, tool_wear_min=tool_wear_min,
                rpm=rpm, air_temp_c=air_temp_c, process_temp_c=process_temp_c), ensure_ascii=False)

        @tool
        def suggest_adjustments(product_type: str, torque: float, tool_wear_min: float,
                                rpm: float | None = None, air_temp_c: float | None = None,
                                process_temp_c: float | None = None) -> str:
            """고장 위험을 낮추는 조정안(공구 교체, 토크 낮추기, 회전수 조정)을 모델로 계산해서 돌려준다.

            Args:
                product_type: 제품 등급 L, M, H 중 하나
                torque: 현재 토크 (Nm)
                tool_wear_min: 현재 공구 사용 시간 (분)
                rpm: 현재 회전수 (rpm), 모르면 비움
                air_temp_c: 주변 온도 (섭씨), 모르면 비움
                process_temp_c: 가공 온도 (섭씨), 모르면 비움
            """
            return json.dumps(agent.suggest_values(
                product_type=product_type, torque=torque, tool_wear_min=tool_wear_min,
                rpm=rpm, air_temp_c=air_temp_c, process_temp_c=process_temp_c), ensure_ascii=False)

        return [diagnose, suggest_adjustments]

    # ------------------------------------------------------------------ LLM
    @property
    def available(self) -> bool:
        return self.provider == "ollama" or bool(self.api_key)

    def _get_llm(self):
        if self._llm is None:
            if self.provider == "ollama":
                from langchain_ollama import ChatOllama
                llm = ChatOllama(model=self.model, base_url=self.ollama_url,
                                 temperature=self.temperature)
            else:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=self.model, api_key=self.api_key,
                                 temperature=self.temperature)
            self._llm = llm.bind_tools(self.tools)
        return self._llm

    def run(self, history: list, question: str) -> tuple[str, list[dict], list]:
        """질문 하나를 처리한다. (답변, 도구 호출 기록, 갱신된 대화 기록) 반환.

        history에는 이전 턴의 메시지(도구 결과 포함)가 들어 있어 후속 질문에 값을 이어 쓸 수 있다.
        """
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        messages = [SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(question)]
        trace = []
        for _ in range(self.max_steps):
            ai = self._get_llm().invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                return self.ensure_assumptions(ai.content, trace), trace, messages[1:]
            for call in ai.tool_calls:
                fn = self.tools_by_name.get(call["name"])
                try:
                    result = fn.invoke(call["args"]) if fn else f"알 수 없는 도구: {call['name']}"
                except Exception as e:  # 잘못된 인자 등은 LLM에게 알려서 고치게 한다
                    result = f"도구 실행 오류: {e}"
                trace.append({"tool": call["name"], "args": call["args"], "result": result})
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return "도구 호출이 너무 많아 답변을 마치지 못했습니다. 질문을 나눠서 다시 해 주세요.", trace, messages[1:]

    # ------------------------------------------------------------------ 검증
    @staticmethod
    def ensure_assumptions(answer: str, trace: list[dict]) -> str:
        """가정한 입력값이 있는데 답변에 언급이 없으면 마지막 줄에 붙인다 (LLM이 빠뜨려도 보장)."""
        if not trace or "가정" in answer:
            return answer
        try:
            assumed = json.loads(trace[-1]["result"]).get("assumed_defaults", "없음")
        except Exception:
            return answer
        if not assumed or assumed == "없음":
            return answer
        if isinstance(assumed, list):
            assumed = ", ".join(assumed) + " 으로 가정함"
        return f"{answer}\n\n_가정: {assumed}_"

    @staticmethod
    def check_numbers(answer: str, trace: list[dict]) -> list[str]:
        """답변의 퍼센트 수치 중 도구 결과에 없는 것을 찾는다 (판정 기준 20%, 50%는 허용)."""
        allowed = {"20", "50"}
        for t in trace:
            for m in re.findall(r"(\d+(?:\.\d+)?)%", str(t["result"])):
                allowed.add(m)
                allowed.add(str(round(float(m))))
        found = re.findall(r"(\d+(?:\.\d+)?)\s*%", answer)
        return sorted({f + "%" for f in found if f not in allowed and str(round(float(f))) not in allowed})
