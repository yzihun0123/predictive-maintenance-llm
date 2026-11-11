"""Streamlit 앱: LLM 정비 상담 에이전트 + 직접 진단 + 모델 성능.

실행: streamlit run app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import MaintenanceAgent
from src.data_loader import AI4IDataLoader
from src.llm_reporter import PROVIDERS, MaintenanceReporter
from src.models import FailurePredictor
from src.rule_checker import FAILURE_DESC_KO, FAILURE_NAMES_KO, FailureRuleChecker

MODEL_PATH = Path("models/predictor.joblib")
K = 273.15  # 켈빈 → 섭씨

MODEL_LABELS = {
    "RandomForest": "랜덤포레스트",
    "Keras DNN (Deep Learning)": "심층 신경망 (Keras)",
    "LogisticRegression": "로지스틱 회귀",
}
FEATURE_LABELS = {
    "type_L": "제품 등급 L", "type_M": "제품 등급 M", "type_H": "제품 등급 H",
    "air_temp": "주변 온도", "process_temp": "가공 온도", "rpm": "회전수",
    "torque": "토크", "tool_wear": "공구 사용 시간",
    "temp_diff": "온도차", "power": "소비 전력", "strain": "공구 부하",
}
PRESETS = {
    "정상 운전": dict(type_="M", air=24.9, proc=35.4, rpm=1500, torque=40.0, wear=50,
                   note="모든 값이 평소 범위 안에 있는 상태"),
    "열이 안 빠지는 상태": dict(type_="L", air=29.4, proc=37.4, rpm=1300, torque=55.0, wear=80,
                        note="주변이 덥고 회전이 느려 가공 열이 쌓이는 상태"),
    "닳은 공구로 무리한 가공": dict(type_="L", air=25.9, proc=36.4, rpm=1350, torque=60.0, wear=210,
                          note="공구를 210분 쓴 상태에서 큰 힘으로 깎는 상태"),
    "전력 부족": dict(type_="H", air=25.4, proc=35.9, rpm=2850, torque=9.0, wear=30,
                  note="빠르게 돌지만 힘이 약해 소비 전력이 기준보다 낮은 상태"),
}

st.set_page_config(page_title="밀링 머신 정비 상담", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.status {padding: 1rem 1.25rem; border-radius: 6px; margin: .5rem 0 1rem 0; border-left: 6px solid;}
.status b {font-size: 1.35rem;}
.status.ok   {background:#eef6ef; border-color:#2e7d32;}
.status.warn {background:#fff6e5; border-color:#ef8f00;}
.status.bad  {background:#fdecec; border-color:#c62828;}
.step {color:#666; font-size:.85rem; font-weight:600; letter-spacing:.05em; margin-bottom:-.5rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="모델을 준비하고 있습니다 (처음 한 번만)")
def get_predictor() -> FailurePredictor:
    if MODEL_PATH.exists():
        try:
            return FailurePredictor.load(MODEL_PATH)
        except Exception:
            pass  # 라이브러리 버전 차이로 로드 실패 시 재학습
    loader = AI4IDataLoader()
    X_train, X_test, y_train, y_test, yt_train, yt_test = loader.split()
    predictor = FailurePredictor().fit(X_train, y_train, yt_train)
    predictor.evaluate(X_test, y_test, yt_test)
    predictor.save(MODEL_PATH)
    return predictor


@st.cache_data(ttl=60, show_spinner=False)
def ollama_running(url: str = "http://localhost:11434/api/tags") -> bool:
    """로컬 Ollama 서버가 켜져 있는지 확인. 배포 환경(Streamlit Cloud)에서는 항상 False."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=0.5):
            return True
    except Exception:
        return False


def get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:
        return os.getenv(name, "")


predictor = get_predictor()

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.subheader("이 서비스는", anchor=False)
    st.caption("작업자가 설비 상태를 말로 물으면, LLM이 학습된 고장 예측 모델(ML/DL)을 직접 호출해 "
               "판정 근거와 조치를 답합니다.")
    st.divider()
    with st.expander("고급 설정"):
        if ollama_running():  # 내 PC에서 Ollama가 켜져 있을 때만 선택지 표시
            provider = st.radio("정비 메모 LLM", ["openai", "ollama"], horizontal=True,
                                format_func={"openai": "OpenAI", "ollama": "Ollama (로컬)"}.get,
                                help="직접 진단 탭의 메모 작성에만 적용됩니다. 정비 상담은 항상 OpenAI를 씁니다.")
        else:  # 배포된 웹 등 Ollama가 없는 환경 → OpenAI 고정
            provider = "openai"
        api_key = ""
        if provider == "openai":
            # Secrets의 키를 입력창 기본값으로 넣으면 브라우저에 노출되므로 넣지 않는다.
            user_key = st.text_input("OpenAI API Key", value="", type="password",
                                     help="비워두면 서버에 저장된 키를 사용합니다.")
            api_key = user_key or get_secret("OPENAI_API_KEY")
            if not user_key and api_key:
                st.caption("서버에 저장된 키 사용 중")
            default_model = get_secret("OPENAI_MODEL") or PROVIDERS["openai"]
        else:
            st.caption("내 PC의 Ollama로 실행합니다 (무료, 1~2분 소요).")
            default_model = PROVIDERS["ollama"]
        llm_model = st.text_input("LLM 모델", value=default_model, key=f"m_{provider}")
        names = list(predictor.binary_models)
        model_name = st.selectbox("예측 모델", names, format_func=MODEL_LABELS.get,
                                  index=names.index(predictor.best_name or "RandomForest"))

st.title("밀링 머신 정비 상담", anchor=False)
st.write("설비 상태를 말로 물어보세요. LLM이 **학습된 고장 예측 모델을 직접 호출**해서 "
         "판정, 근거, 할 일을 알려줍니다.")

with st.expander("처음 보시는 분께: 이 서비스가 하는 일"):
    st.markdown(
        "- **정비 상담**: \"L등급인데 토크 60에 공구 210분 썼어, 괜찮아?\"처럼 물으면, LLM이 질문에서 값을 "
        "뽑아 고장 예측 모델을 호출하고 그 결과를 근거로 답합니다. 해결책을 물으면 조정안(공구 교체, "
        "토크 낮추기 등)도 모델로 계산해서 비교합니다.\n"
        "- **직접 진단**: 슬라이더로 값을 바꿔 가며 모델 결과를 바로 확인하고, 정비 메모를 만듭니다.\n"
        "- **모델 성능**: 예측 모델(랜덤포레스트, Keras 심층 신경망 등)의 테스트 결과입니다.\n\n"
        "숫자(확률, 판정)는 항상 모델이 계산하고, LLM은 그 숫자로만 설명합니다.  \n"
        "학습 데이터: AI4I 2020 (UCI 공개 데이터, 밀링 머신 10,000건, 고장 비율 3.4%)"
    )
    st.dataframe(
        pd.DataFrame({"고장 유형": FAILURE_NAMES_KO.values(), "설명": FAILURE_DESC_KO.values()}),
        hide_index=True, use_container_width=True,
    )

tab_chat, tab_demo, tab_perf = st.tabs(["정비 상담", "직접 진단", "모델 성능"])

# ---------------- 정비 상담 (LLM 에이전트) ----------------
EXAMPLES = [
    "L등급 제품인데 토크 60Nm, 공구 210분 썼어. 계속 돌려도 돼?",
    "그럼 어떻게 하면 돼? 공구 교체 말고 다른 방법도 있어?",
    "M등급, 회전수 1300, 토크 55, 공구 80분인데 주변 온도가 29도로 더워. 괜찮아?",
]

with tab_chat:
    # 도구 호출(Tool Calling)이 안정적인 OpenAI로만 상담한다. (로컬 소형 모델은 도구 호출을 제대로 못 함)
    chat_key = api_key or get_secret("OPENAI_API_KEY")
    chat_model = (llm_model if provider == "openai" else None) or get_secret("OPENAI_MODEL") \
        or PROVIDERS["openai"]
    agent = MaintenanceAgent(predictor, provider="openai", api_key=chat_key or None, model=chat_model)

    if "chat" not in st.session_state:
        st.session_state.chat = []      # 화면 표시용: {"role", "content", "trace", "flags"}
        st.session_state.history = []   # LLM 대화 기록 (도구 결과 포함)

    head_l, head_r = st.columns([4, 1])
    with head_r:
        if st.button("대화 지우기", use_container_width=True):
            st.session_state.chat, st.session_state.history = [], []
            st.rerun()

    if not agent.available:
        st.info("정비 상담은 LLM이 필요합니다. 고급 설정에 OpenAI API 키를 넣어 주세요.")

    if not st.session_state.chat:
        st.markdown("**이렇게 물어보세요**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}", disabled=not agent.available):
                st.session_state.pending = ex
                st.rerun()

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("flags"):
                st.warning("도구 결과에 없는 수치가 답변에 있습니다: " + ", ".join(msg["flags"])
                           + ". 아래 모델 호출 내역의 수치를 기준으로 확인하세요.")
            if msg.get("trace"):
                with st.expander(f"모델 호출 내역 ({len(msg['trace'])}회)"):
                    for t in msg["trace"]:
                        st.markdown(f"`{t['tool']}` 입력: `{json.dumps(t['args'], ensure_ascii=False)}`")
                        try:
                            st.json(json.loads(t["result"]), expanded=False)
                        except Exception:
                            st.code(str(t["result"]))

    question = st.chat_input("설비 상태를 입력하세요 (예: H등급, 토크 40, 공구 120분)",
                             disabled=not agent.available)
    question = question or st.session_state.pop("pending", None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.spinner("모델을 호출하며 답변을 만드는 중..."):
            try:
                answer, trace, history = agent.run(st.session_state.history, question)
                st.session_state.history = history
                st.session_state.chat.append({
                    "role": "assistant", "content": answer, "trace": trace,
                    "flags": MaintenanceAgent.check_numbers(answer, trace)})
            except Exception as e:
                st.session_state.chat.append({"role": "assistant",
                                              "content": f"LLM 호출에 실패했습니다: {e}"})
        st.rerun()

# ---------------- 직접 진단 ----------------
with tab_demo:
    preset = st.radio("예시 상황", list(PRESETS), horizontal=True)
    p = PRESETS[preset]
    st.caption(PRESETS[preset]["note"])
    st.markdown('<p class="step">STEP 1</p>', unsafe_allow_html=True)
    st.subheader("설비 상태", anchor=False)
    c1, c2, c3 = st.columns(3)
    k = preset  # 상황을 바꾸면 슬라이더 값도 초기화되도록 key에 포함
    with c1:
        type_ = st.selectbox("제품 등급", ["L", "M", "H"], index="LMH".index(p["type_"]), key=f"t{k}",
                             help="L·M·H 순으로 품질 기준이 높아지고, 견딜 수 있는 공구 부하도 커집니다.")
        rpm = st.slider("회전수 (rpm)", 1150, 2900, p["rpm"], 10, key=f"r{k}")
    with c2:
        air_c = st.slider("주변 온도 (°C)", 21.9, 31.9, p["air"], 0.1, key=f"a{k}")
        torque = st.slider("토크 (Nm)", 3.0, 77.0, p["torque"], 0.5, key=f"q{k}",
                           help="공구를 돌리는 힘. 클수록 세게 깎는 것")
    with c3:
        proc_c = st.slider("가공 온도 (°C)", 31.9, 40.9, p["proc"], 0.1, key=f"p{k}")
        wear = st.slider("공구 사용 시간 (분)", 0, 255, p["wear"], 1, key=f"w{k}",
                         help="공구를 교체한 뒤 지금까지 사용한 시간")

    X_row = AI4IDataLoader.make_single(type_, air_c + K, proc_c + K, rpm, torque, wear)
    sensor = X_row.iloc[0].to_dict()
    pred = predictor.predict_one(X_row, model_name)
    findings = FailureRuleChecker().check(sensor)
    prob = pred["failure_probability"]

    st.markdown('<p class="step">STEP 2</p>', unsafe_allow_html=True)
    st.subheader("진단 결과", anchor=False)
    label = MaintenanceReporter.judge(pred, findings)  # 정비 메모와 같은 기준으로 판정
    cls, msg = {
        "위험": ("bad", "곧 고장이 날 가능성이 높습니다. 바로 점검이 필요합니다."),
        "주의": ("warn", "고장 징후가 보입니다. 가까운 시일 안에 점검하세요."),
        "정상": ("ok", "고장 징후가 거의 없습니다."),
    }[label]
    st.markdown(f'<div class="status {cls}"><b>{label} · 고장 확률 {prob:.0%}</b><br>{msg}</div>',
                unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("온도차 (가공 − 주변)", f"{sensor['temp_diff']:.1f} °C",
              help="작을수록 열이 잘 안 빠집니다. 8.6°C 미만이면 주의")
    m2.metric("소비 전력", f"{sensor['power']:,.0f} W",
              help="토크 × 회전 속도. 3,500~9,000W가 정상 범위")
    m3.metric("공구 부하", f"{sensor['strain']:,.0f}",
              help="공구 사용 시간 × 토크. 닳은 공구에 걸리는 부담을 나타냄")

    left, right = st.columns([1, 1.7])
    with left:
        st.markdown("**원인별 고장 가능성** (모델 예측)")
        st.dataframe(
            pd.DataFrame({
                "고장 유형": [FAILURE_NAMES_KO[c] for c in pred["failure_type_probabilities"]],
                "가능성": list(pred["failure_type_probabilities"].values()),
            }).sort_values("가능성", ascending=False),
            hide_index=True, use_container_width=True,
            column_config={"가능성": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0)},
        )
    with right:
        st.markdown("**기준값 비교**")
        rule_df = pd.DataFrame({
            "고장 유형": [f["name"] for f in findings],
            "판정": ["기준 초과" if f["triggered"] else "정상" for f in findings],
            "현재 값": [f["value"] for f in findings],
            "위험 기준": [f["criterion"] for f in findings],
        })
        st.dataframe(
            rule_df.style.map(
                lambda v: "color:#c62828;font-weight:600" if v == "기준 초과" else "color:#2e7d32",
                subset=["판정"]),
            hide_index=True, use_container_width=True,
            column_config={
                "고장 유형": st.column_config.TextColumn(width="small"),
                "판정": st.column_config.TextColumn(width="small"),
                "현재 값": st.column_config.TextColumn(width="small"),
                "위험 기준": st.column_config.TextColumn(width="medium"),
            },
        )

    st.markdown('<p class="step">STEP 3</p>', unsafe_allow_html=True)
    st.subheader("정비 메모", anchor=False)
    st.caption("위 진단 결과와 기준값 비교 내용을 LLM에 넘겨, 현장에서 바로 읽을 수 있는 메모로 정리합니다.")
    if st.button("정비 메모 작성", type="primary"):
        reporter = MaintenanceReporter(provider=provider, api_key=api_key or None, model=llm_model)
        with st.spinner("작성 중..."):
            try:
                report = reporter.generate(sensor, pred, findings)
                with st.container(border=True):
                    st.markdown(report)
            except Exception as e:
                st.error(f"LLM 호출에 실패했습니다: {e}")
        with st.expander("LLM에 넘긴 데이터 보기"):
            st.code(MaintenanceReporter.build_prompt(sensor, pred, findings), language="json")

# ---------------- 모델 성능 ----------------
with tab_perf:
    m = predictor.metrics
    if not m:
        st.info("성능 지표가 없습니다. `python train.py`를 먼저 실행하세요.")
    else:
        best = m["best_model"]
        (tn, fp), (fn, tp) = m["binary"][best]["confusion_matrix"]
        st.subheader("한눈에 보기", anchor=False)
        st.write(f"학습에 쓰지 않은 테스트 데이터 **{tn + fp + fn + tp:,}건**으로 평가했습니다 "
                 f"(대표 모델: {MODEL_LABELS[best]}).")
        a, b, c = st.columns(3)
        a.metric("실제 고장 중 잡아낸 건수", f"{tp} / {tp + fn}건")
        b.metric("놓친 고장", f"{fn}건")
        c.metric("정상인데 잘못 울린 경보", f"{fp}건")

        st.subheader("모델 비교", anchor=False)
        df = pd.DataFrame(m["binary"]).T.drop(columns="confusion_matrix").astype(float)
        df.index = [MODEL_LABELS[i] for i in df.index]
        df.columns = ["정확도", "정밀도", "재현율", "F1", "AUC"]
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
        st.caption("정밀도: 경보가 울렸을 때 실제 고장인 비율 · 재현율: 실제 고장 중 잡아낸 비율 · "
                   "F1: 둘의 균형. 고장이 3.4%뿐이라 모두 '정상'이라고만 해도 정확도가 96.6%가 나오므로 "
                   "정확도보다 F1과 재현율을 기준으로 봤습니다.")

        st.subheader("심층 신경망 학습 과정", anchor=False)
        hist = predictor.dl_history()
        st.line_chart(pd.DataFrame({"학습 손실": hist["loss"], "검증 손실": hist["val_loss"]}),
                      x_label="epoch", y_label="loss")
        st.caption(f"총 {len(hist['loss'])} epoch 학습. 검증 손실이 더 이상 줄지 않으면 "
                   "EarlyStopping으로 멈추고, 가장 좋았던 시점의 가중치를 사용합니다. "
                   "검증 손실이 더 낮은 것은 학습 때만 Dropout과 고장 가중치(class_weight)가 "
                   "적용되기 때문이며, 과적합 신호가 아닙니다.")

        st.subheader("고장 유형별 성능 (랜덤포레스트)", anchor=False)
        ft = pd.DataFrame(m["failure_types"]).T
        ft.index = [FAILURE_NAMES_KO[i] for i in ft.index]
        ft = ft.rename(columns={"support": "테스트 건수", "recall": "재현율",
                                "precision": "정밀도", "f1": "F1"})
        st.dataframe(ft.style.format({"테스트 건수": "{:.0f}", "재현율": "{:.2f}",
                                      "정밀도": "{:.2f}", "F1": "{:.2f}"}),
                     use_container_width=True)
        st.caption("공구 마모와 원인 불명 고장은 거의 무작위로 발생해서 센서값만으로는 예측하지 못했습니다.")

        st.subheader("모델이 중요하게 본 값", anchor=False)
        fi = predictor.feature_importance()
        st.dataframe(
            pd.DataFrame({"항목": [FEATURE_LABELS.get(i, i) for i in fi.index], "중요도": fi.values}),
            hide_index=True, use_container_width=True,
            column_config={"중요도": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=float(fi.max()))},
        )
        st.caption("회전수·토크 다음으로 직접 만든 '소비 전력'이 3위입니다. 공구 부하와 온도차도 "
                   "원래 온도값보다 높게 평가돼, 고장 원리에 맞춰 만든 변수가 도움이 됐습니다.")
