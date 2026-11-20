# streamlit_app.py
import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# =============================
# 0. 기본 설정
# =============================

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="EmotionCare - 표정 기반 음악 케어 시스템",
    page_icon="🎧",
    layout="wide",
)

# =============================
# 1. 사이드바 & 헤더
# =============================

st.sidebar.title("EmotionCare Console")
st.sidebar.markdown(
    """
**사용 방법**

1. 웹캠으로 얼굴을 정면으로 바라보고 캡처합니다.  
2. `감정 분석 & 음악 추천` 버튼을 누릅니다.  
3. 오른쪽 패널에서 감정 결과, 확률 그래프, 추천 플레이리스트를 확인합니다.

> 조명은 너무 어둡지 않게, 얼굴이 화면 중앙에 오도록 맞춰주세요.
"""
)

st.title("EmotionCare 🎧")
st.markdown(
    "캡처한 얼굴 이미지의 표정을 분류하고, "
    "상태에 맞는 **음악 플레이리스트**를 추천해주는 시스템입니다."
)

# =============================
# 2. 메인 레이아웃
# =============================

col_left, col_right = st.columns([1.2, 1], gap="large")

# ---------- 왼쪽: 카메라 ----------
with col_left:
    st.subheader("1️⃣ 얼굴 캡처")

    img_file = st.camera_input("웹캠으로 얼굴을 찍어주세요.", key="camera")

    analyze_clicked = False
    if img_file is not None:
        st.success("사진이 캡처되었습니다. 아래 버튼으로 분석을 진행하세요.")
        analyze_clicked = st.button("🧠 감정 분석 & 음악 추천", type="primary")
    else:
        st.info("카메라가 보이지 않으면 브라우저의 카메라 권한을 확인해주세요.")

# ---------- 오른쪽: 결과 ----------
with col_right:
    st.subheader("2️⃣ 분석 결과")

    if img_file is None:
        st.info("왼쪽에서 얼굴 사진을 캡처한 후 분석을 진행해주세요.")
    elif analyze_clicked:
        # API 호출
        with st.spinner("감정을 분석하고 음악을 추천하는 중입니다..."):
            try:
                bytes_data = img_file.getvalue()
                files = {"file": ("frame.jpg", bytes_data, "image/jpeg")}
                res = requests.post(API_URL, files=files)
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했습니다: {e}")
                st.stop()

            if res.status_code != 200:
                try:
                    detail = res.json().get("detail", "")
                except Exception:
                    detail = ""
                st.error(f"에러 {res.status_code}: {detail}")
                st.stop()

            data = res.json()

        # ---------- 결과 파싱 ----------
        emotion_label = data["emotion_label"]
        prob_dict = data["probabilities"]
        playlist = data["playlist"]
        face_b64 = data.get("face_image")

        # 상단: 감정 요약
        st.markdown("#### 🔍 감정 요약")
        st.markdown(
            f"**예측 감정:** :orange[`{emotion_label}`]  "
        )

        # 중앙: 검출된 얼굴 이미지 + 확률 그래프
        face_col, chart_col = st.columns([1, 1.5])

        with face_col:
            st.markdown("##### 🎭 검출된 얼굴")
            if face_b64:
                face_bytes = base64.b64decode(face_b64)
                face_img = Image.open(io.BytesIO(face_bytes))
                st.image(face_img, caption="YOLO Face Detection 결과", use_column_width=True)
            else:
                st.warning("검출된 얼굴 이미지를 가져오지 못했습니다.")

        with chart_col:
            st.markdown("##### 📊 감정 확률 분포")

            df = pd.DataFrame(
                {
                    "emotion": list(prob_dict.keys()),
                    "probability": [prob_dict[k] for k in prob_dict.keys()],
                }
            )
            df = df.sort_values("probability", ascending=False)

            # 막대 그래프
            fig, ax = plt.subplots()
            ax.bar(df["emotion"], df["probability"])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Probability")
            ax.set_title("Emotion Probabilities")
            for idx, row in df.iterrows():
                ax.text(
                    row["emotion"],
                    row["probability"] + 0.01,
                    f"{row['probability']:.2f}",
                    ha="center",
                )
            st.pyplot(fig)

            # 표 형태로도 같이 보여주기
            st.dataframe(
                df.set_index("emotion").style.format({"probability": "{:.3f}"}),
                use_container_width=True,
            )

        # 하단: 추천 플레이리스트
        st.markdown("#### 🎵 맞춤형 음악 케어 플랜")

        if not playlist:
            st.warning("해당 감정에 대한 추천 플레이리스트가 없습니다.")
        else:
            for track in playlist:
                # track은 dict 형태 (FastAPI가 Pydantic 모델을 JSON으로 변환한 것)
                title = track.get("title")
                artist = track.get("artist")
                mood = track.get("mood")
                url = track.get("url")

                with st.container():
                    st.markdown(
                        f"**{title}**  \n"
                        f"- 아티스트: `{artist}`  \n"
                        f"- 용도/분위기: _{mood}_  \n"
                        f"- 링크: [바로 듣기]({url})"
                    )
                    st.markdown("---")
