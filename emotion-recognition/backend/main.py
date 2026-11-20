# backend/main.py
import base64
import io
import os
from typing import List, Dict, Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# =============================
# 0. 기본 설정
# =============================

# 프로젝트 루트 기준 경로들
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 학습된 감정 분류 모델 가중치 (학습 체크포인트)
MODEL_WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "ck_only_results",
    "ck_only_stratified_split_resnet50_ep200_pat10.weights.h5",
)

# YOLO face 가중치 (별도로 준비하는 가중치)
YOLO_WEIGHT_PATH = os.path.join(BASE_DIR, "weights", "yolov8n-face.pt")

IMG_SIZE = (96, 96)
NEW_CLASS_NAMES = ["anger", "happy", "sadness", "nervousness"]

# =============================
# 1. FastAPI 기본 세팅
# =============================

app = FastAPI(title="EmotionCare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# 2. 감정 모델 빌드 & 로딩
# =============================

def build_emotion_model(input_shape=(96, 96, 3), num_classes=4):
    """학습 때 사용한 ResNet50 백본 구조 재현."""
    inputs = layers.Input(shape=input_shape)

    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_emotion")
    return model


print("[INFO] Building emotion model (ResNet50)...")
emotion_model = build_emotion_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=len(NEW_CLASS_NAMES),
)
# compile은 predict에 꼭 필요하진 않지만, 일부 TF 버전에서 warning 방지
emotion_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("[INFO] Loading emotion model weights from:", MODEL_WEIGHTS_PATH)
emotion_model.load_weights(MODEL_WEIGHTS_PATH)
print("[INFO] Emotion model weights loaded.")


print("[INFO] Loading YOLO face model...")
yolo_face = YOLO(YOLO_WEIGHT_PATH)
print("[INFO] YOLO face model loaded from:", YOLO_WEIGHT_PATH)

# =============================
# 3. Pydantic 스키마
# =============================

class TrackInfo(BaseModel):
    title: str
    artist: str
    mood: str     # 예: "calming", "focus", "uplifting"
    url: str


class EmotionPrediction(BaseModel):
    emotion_id: int
    emotion_label: str
    probabilities: Dict[str, float]
    playlist: List[TrackInfo]
    face_image: Optional[str] = None  # base64-encoded JPEG


# =============================
# 4. 음악 추천 DB (클래스별 정적 매핑)
# =============================

PLAYLIST_DB: Dict[str, List[TrackInfo]] = {
    "anger": [
        TrackInfo(
            title="Weightless",
            artist="Marconi Union",
            mood="calming / anxiety reduction",
            url="https://www.youtube.com/watch?v=UfcAVejslrU",
        ),
        TrackInfo(
            title="Nuvole Bianche",
            artist="Ludovico Einaudi",
            mood="emotional release / piano",
            url="https://www.youtube.com/watch?v=kcihcYEOeic",
        ),
        TrackInfo(
            title="Autumn Leaves (Instrumental Jazz)",
            artist="Cafe Jazz Trio",
            mood="relaxing / grounding",
            url="https://www.youtube.com/watch?v=Dx5qFachd3A",
        ),
        TrackInfo(
            title="Deep Focus Ambient",
            artist="Longform Ambient Mix",
            mood="cool-down / deep breathing",
            url="https://www.youtube.com/watch?v=hHW1oY26kxQ",
        ),
    ],
    "happy": [
        TrackInfo(
            title="Good Life",
            artist="OneRepublic",
            mood="feel-good pop",
            url="https://www.youtube.com/watch?v=jZhQOvvV45w",
        ),
        TrackInfo(
            title="Sugar",
            artist="Maroon 5",
            mood="upbeat / energetic",
            url="https://www.youtube.com/watch?v=09R8_2nJtjg",
        ),
        TrackInfo(
            title="Summer",
            artist="Calvin Harris",
            mood="EDM / driving",
            url="https://www.youtube.com/watch?v=ebXbLfLACGM",
        ),
        TrackInfo(
            title="Happy (Instrumental Jazz Cover)",
            artist="Jazz Lounge",
            mood="light mood / background",
            url="https://www.youtube.com/watch?v=OFWSf8J5RTs",
        ),
    ],
    "sadness": [
        TrackInfo(
            title="River Flows in You",
            artist="Yiruma",
            mood="soft piano / emotional support",
            url="https://www.youtube.com/watch?v=7maJOI3QMu0",
        ),
        TrackInfo(
            title="Clair de Lune",
            artist="Debussy",
            mood="soothing classical",
            url="https://www.youtube.com/watch?v=4fvo_iOuSck",
        ),
        TrackInfo(
            title="Rainy Night Lo-Fi",
            artist="Lo-Fi Girl",
            mood="comfort / journaling",
            url="https://www.youtube.com/watch?v=MVPTGNGiI-4",
        ),
        TrackInfo(
            title="Acoustic Covers for Study",
            artist="Acoustic Chill",
            mood="warm acoustic",
            url="https://www.youtube.com/watch?v=2S9vF7G94Xg",
        ),
    ],
    "nervousness": [
        TrackInfo(
            title="Deep Focus",
            artist="Alpha Waves",
            mood="concentration / exam mode",
            url="https://www.youtube.com/watch?v=hHW1oY26kxQ",
        ),
        TrackInfo(
            title="Lo-Fi Beats for Study",
            artist="Lo-Fi Study Mix",
            mood="light focus / background",
            url="https://www.youtube.com/watch?v=jfKfPfyJRdk",
        ),
        TrackInfo(
            title="Spa & Mindfulness",
            artist="Relaxing Zen Music",
            mood="breathing / mindfulness",
            url="https://www.youtube.com/watch?v=1ZYbU82GVz4",
        ),
        TrackInfo(
            title="Soft Piano for Sleep",
            artist="Piano Relax",
            mood="sleep preparation",
            url="https://www.youtube.com/watch?v=1fueZCTYkpA",
        ),
    ],
}

# =============================
# 5. 유틸 함수
# =============================

def read_imagefile_to_cv2(data: bytes) -> np.ndarray:
    """바이너리 이미지 데이터를 OpenCV BGR 이미지로 변환."""
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return img  # BGR


def detect_face_and_crop_bgr(bgr_img: np.ndarray) -> np.ndarray:
    """
    YOLO face로 얼굴 1개 검출 후 crop된 BGR 이미지를 반환.
    - 가장 confidence가 높은 얼굴만 사용.
    - YOLO가 실패하면 이미지 중앙부를 얼굴로 가정하는 fallback 사용.
    """
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # conf를 낮춰서 얼굴을 더 넉넉하게 잡도록
    results = yolo_face(rgb_img, verbose=False, conf=0.2)

    face_bgr = None

    if results and getattr(results[0], "boxes", None) is not None:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            scores = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            best_idx = int(np.argmax(scores))
            x1, y1, x2, y2 = xyxy[best_idx]

            h, w, _ = bgr_img.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_bgr = bgr_img[y1:y2, x1:x2]

    # fallback: 중앙 정사각형 크롭
    if face_bgr is None or face_bgr.size == 0:
        h, w, _ = bgr_img.shape
        side = int(min(h, w) * 0.6)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        face_bgr = bgr_img[y1:y2, x1:x2]

    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("얼굴을 찾지 못했습니다.")

    return face_bgr


def preprocess_for_emotion_model(face_bgr: np.ndarray) -> np.ndarray:
    """
    감정 모델 입력 전처리:
    - BGR -> Gray (CK+ 특성에 맞춤)
    - 히스토그램 평활화로 명암 대비 보정
    - 다시 3채널로 복제 (ResNet50은 3채널 입력 기대)
    - 96x96 리사이즈
    - 0~1 스케일링
    """
    # BGR -> Grayscale
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # 명암 대비 보정 (얼굴 부분이 더 또렷하게)
    face_gray = cv2.equalizeHist(face_gray)

    # 리사이즈
    face_gray = cv2.resize(face_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 0~1 스케일링
    face_gray = face_gray.astype("float32") / 255.0  # (H, W)

    # 1채널 -> 3채널 복제 (H, W, 3)
    face_3ch = np.stack([face_gray, face_gray, face_gray], axis=-1)

    # 배치 차원 추가: (1, H, W, 3)
    face_3ch = np.expand_dims(face_3ch, axis=0)

    return face_3ch



def predict_emotion(face_tensor: np.ndarray) -> EmotionPrediction:
    preds = emotion_model.predict(face_tensor, verbose=0)
    probs = preds[0]
    emotion_idx = int(np.argmax(probs))
    emotion_label = NEW_CLASS_NAMES[emotion_idx]

    prob_dict = {
        NEW_CLASS_NAMES[i]: float(probs[i]) for i in range(len(NEW_CLASS_NAMES))
    }

    playlist = PLAYLIST_DB.get(emotion_label, [])

    return EmotionPrediction(
        emotion_id=emotion_idx,
        emotion_label=emotion_label,
        probabilities=prob_dict,
        playlist=playlist,
    )


def encode_face_to_base64(face_bgr: np.ndarray) -> str:
    """크롭된 얼굴 이미지를 JPEG -> base64 문자열로 인코딩."""
    _, jpeg = cv2.imencode(".jpg", face_bgr)
    b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
    return b64


# =============================
# 6. 엔드포인트
# =============================

@app.get("/")
def root():
    return {"message": "EmotionCare API. Use POST /predict with an image."}


@app.post("/predict", response_model=EmotionPrediction)
async def predict(file: UploadFile = File(...)):
    """
    multipart/form-data로 이미지 파일을 받아
    - YOLO face로 얼굴 검출 / 크롭
    - 감정 분류
    - 추천 플레이리스트 + 크롭된 얼굴(base64) 반환
    """
    try:
        contents = await file.read()
        bgr_img = read_imagefile_to_cv2(contents)
        face_bgr = detect_face_and_crop_bgr(bgr_img)
        face_tensor = preprocess_for_emotion_model(face_bgr)

        result = predict_emotion(face_tensor)
        result.face_image = encode_face_to_base64(face_bgr)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("[ERROR]", e)
        raise HTTPException(status_code=500, detail="Internal server error")
