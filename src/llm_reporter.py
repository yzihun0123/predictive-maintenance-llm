"""LangChain으로 정비 메모를 생성하는 클래스 (OpenAI 또는 로컬 Ollama 선택).

체인 구성 (LCEL): ChatPromptTemplate | LLM | StrOutputParser

설계 원칙: 판정(정상/주의/위험)은 코드가 정하고, LLM은 그 판정의 근거와 조치만 설명한다.
LLM이 판정을 바꿔 쓰면 verify_verdict()가 바로잡는다 (환각 방지).
"""
from __future__ import annotations

import json
import os
import re

from .rule_checker import FAILURE_NAMES_KO

SYSTEM_PROMPT = """너는 밀링 머신을 15년째 관리해 온 설비 보전 담당자다.
교대 근무자에게 넘기는 점검 메모를 쓴다. 읽는 사람은 바쁜 현장 작업자다.

반드시 지킬 것:
- 판정은 입력의 "판정" 값을 그대로 쓴다. 절대 바꾸지 않는다.
- "원인"에는 "기준_초과_항목"에 있는 것만 쓴다. "기준_이내_항목"은 원인으로 쓰지 않는다.
- 기준_초과_항목이 없으면 원인은 "기준 초과 항목 없음" 한 줄만 쓴다.
- 판정이 정상이면 조치는 "현 상태로 운전 유지, 정기 점검 주기대로 확인" 수준으로 1~2줄만 쓴다.
- 원인 한 줄마다 "현재값 → 위험기준" 형식으로 수치를 함께 쓴다. (예: 과부하 — 부하 12,600 → L등급 11,000 초과)
- 조치는 원인과 연결해서 구체적으로 쓴다. (예: 공구 교체, 토크 설정 낮춤)
- 입력 JSON의 키 이름(기준_초과_항목 등)을 그대로 쓰지 않는다.
- 주어진 수치만 근거로 쓴다. 부품명, 과거 이력처럼 주어지지 않은 정보는 만들지 않는다.

문체:
- 인사말, 맺음말, 감탄, 이모지, "~해 보세요" 같은 표현 금지.
- 개조식(~함, ~필요, ~권장)으로 짧게. 전체 12줄 이내. 수치에는 단위를 붙인다.

형식 (그대로 따를 것, 제목 크기 키우지 말 것):
**판정: (입력의 판정)** — 한 줄 근거

**원인**
- ...

**조치** (급한 순서)
1. ...

**다음 점검 때 확인**
- ...
"""


PROVIDERS = {"openai": "gpt-4o-mini", "ollama": "gemma2"}  # 공급자별 기본 모델


class MaintenanceReporter:
    """예측 결과 + 규칙 점검 결과 → 판정 확정 → LangChain 체인 → 판정 검증."""

    def __init__(self, provider: str = "openai", api_key: str | None = None,
                 model: str | None = None, temperature: float = 0.2,
                 ollama_url: str = "http://localhost:11434"):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or (os.getenv("OPENAI_MODEL") if provider == "openai" else None) \
            or PROVIDERS[provider]
        self.temperature = temperature
        self.ollama_url = ollama_url
        self._chain = None

    @property
    def available(self) -> bool:
        # Ollama는 키가 필요 없다 (로컬 서버가 꺼져 있으면 호출 시 에러)
        return self.provider == "ollama" or bool(self.api_key)

    def _build_llm(self):
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=self.model, base_url=self.ollama_url,
                              temperature=self.temperature)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model, api_key=self.api_key, temperature=self.temperature)

    def get_chain(self):
        """LCEL 체인: 프롬프트 템플릿 | LLM | 문자열 출력 파서."""
        if self._chain is None:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{diagnosis}"),  # 진단 JSON은 변수로 넣어 중괄호 충돌을 피한다
            ])
            self._chain = prompt | self._build_llm() | StrOutputParser()
        return self._chain

    # ---------- 판정 (코드가 결정) ----------
    @staticmethod
    def judge(prediction: dict, rule_findings: list[dict]) -> str:
        """고장 확률 50% 이상이면 위험, 20% 이상이거나 기준 초과 항목이 있으면 주의."""
        p = prediction["failure_probability"]
        if p >= 0.5:
            return "위험"
        if p >= 0.2 or any(f["triggered"] for f in rule_findings):
            return "주의"
        return "정상"

    # ---------- 프롬프트 ----------
    @classmethod
    def build_prompt(cls, sensor: dict, prediction: dict, rule_findings: list[dict]) -> str:
        over = [f for f in rule_findings if f["triggered"]]
        within = [f for f in rule_findings if not f["triggered"]]
        payload = {
            "판정": cls.judge(prediction, rule_findings),
            "고장_확률": f"{prediction['failure_probability']:.1%}",
            "기준_초과_항목": [
                {"유형": f["name"], "현재값": f["value"], "위험기준": f["criterion"]} for f in over
            ] or "없음",
            "기준_이내_항목": [f["name"] for f in within] or "없음",
            "모델이_본_유형별_가능성": {
                FAILURE_NAMES_KO[k]: f"{v:.1%}"
                for k, v in prediction["failure_type_probabilities"].items()
            },
            "센서값": {
                "제품등급": sensor["type"],
                "주변온도_C": round(sensor["air_temp"] - 273.15, 1),
                "가공온도_C": round(sensor["process_temp"] - 273.15, 1),
                "온도차_C": round(sensor["temp_diff"], 1),
                "회전수_rpm": round(sensor["rpm"]),
                "토크_Nm": round(sensor["torque"], 1),
                "소비전력_W": round(sensor["power"]),
                "공구사용시간_min": round(sensor["tool_wear"]),
                "공구부하": round(sensor["strain"]),
            },
        }
        return "아래 진단 결과로 점검 메모를 작성하라.\n\n" + json.dumps(
            payload, ensure_ascii=False, indent=2)

    # ---------- 검증 ----------
    @staticmethod
    def verdict_line(level: str, prediction: dict, rule_findings: list[dict]) -> str:
        n = sum(f["triggered"] for f in rule_findings)
        basis = f"기준 초과 {n}개 항목" if n else "기준 초과 항목 없음"
        return f"**판정: {level}** — 고장 확률 {prediction['failure_probability']:.1%}, {basis}"

    @staticmethod
    def verify_verdict(report: str, correct_line: str, level: str) -> tuple[str, bool]:
        """LLM이 쓴 판정이 코드의 판정과 다르면 판정 줄 전체를 바꾼다. (메모, 수정 여부) 반환."""
        m = re.search(r"판정\s*[:：]\s*\**\s*(정상|주의|위험)", report)
        if m and m.group(1) == level:
            return report, False
        if m:  # 판정 단어가 들어 있는 줄 전체를 교체 (틀린 근거 문장까지 제거)
            start = report.rfind("\n", 0, m.start()) + 1
            end = report.find("\n", m.end())
            end = len(report) if end == -1 else end
            return report[:start] + correct_line + report[end:], True
        return correct_line + "\n\n" + report, True

    # ---------- 생성 ----------
    def generate(self, sensor: dict, prediction: dict, rule_findings: list[dict]) -> str:
        level = self.judge(prediction, rule_findings)
        if not self.available:
            return self.fallback_report(prediction, rule_findings)
        raw = self.get_chain().invoke(
            {"diagnosis": self.build_prompt(sensor, prediction, rule_findings)})
        report, corrected = self.verify_verdict(
            raw, self.verdict_line(level, prediction, rule_findings), level)
        if corrected:
            report += "\n\n_※ LLM이 쓴 판정이 진단 결과와 달라 자동으로 바로잡았습니다._"
        return report

    @classmethod
    def fallback_report(cls, prediction: dict, rule_findings: list[dict]) -> str:
        """API 키가 없을 때 쓰는 기본 양식 (LLM 미사용)."""
        level = cls.judge(prediction, rule_findings)
        risks = [f"- {f['name']}: {f['detail']}" for f in rule_findings if f["triggered"]]
        return "\n".join([
            "_API 키가 없어 LLM 대신 기본 양식으로 표시합니다._",
            "", cls.verdict_line(level, prediction, rule_findings),
            "", "**원인**", *(risks or ["- 기준 초과 항목 없음"]),
        ])
