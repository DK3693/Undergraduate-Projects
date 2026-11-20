# 모델·API·실행 조건

## 데이터와 라벨

학습 코드는 CK+ 계열의 폴더별 얼굴 표정 데이터셋을 입력으로 사용합니다. 원본 자료의 이용 조건에 따라 별도로 확보합니다.

| 출력 | 원본 라벨 |
|---|---|
| anger | anger, contempt, disgust |
| happy | happy |
| sadness | sadness |
| nervousness | fear, surprise |

학습/검증/시험 비율은 클래스별 약 70/15/15이며, 정수 처리 후 남은 이미지는 시험 세트에 배정합니다. 분할 단위는 이미지로, 인물 독립 분할이 아닙니다. 증강은 학습 세트에만 적용합니다. 공개 코드에는 클래스 가중치와 EarlyStopping, ModelCheckpoint, 혼동 행렬 출력이 포함됩니다.

## 모델 파일

| 경로 | 용도 |
|---|---|
| `weights/yolov8n-face.pt` | 얼굴 검출 모델 |
| `ck_only_results/ck_only_stratified_split_resnet50_ep200_pat10.weights.h5` | 추론 서버가 읽는 표정 분류 가중치 |
| `data/ck/` 또는 `CK_ROOT` | 학습용 클래스별 이미지 폴더 |

얼굴 검출 가중치는 사용한 배포자의 이용 조건을 확인해 준비합니다. ImageNet 사전 학습 가중치는 Keras가 최초 실행 시 내려받을 수 있습니다. 필요한 패키지를 나열한 requirements이며, 원래 학습 환경을 고정한 lock 파일은 아닙니다.

## API 계약

`POST /predict`는 `multipart/form-data`의 `file` 필드로 이미지 한 장을 받습니다.

| 응답 필드 | 의미 |
|---|---|
| `emotion_id` | 0~3 클래스 인덱스 |
| `emotion_label` | 클래스 이름 |
| `probabilities` | 클래스별 모델 출력 확률 |
| `playlist` | 제목·아티스트·분류·링크 목록 |
| `face_image` | 분석에 사용한 크롭 이미지의 Base64 JPEG |

얼굴 검출 결과가 없으면 중앙 영역을 크롭하는 대체 경로가 있습니다. API 응답의 분류값만으로 얼굴 검출 성공 여부를 단정하지 않습니다.

## 결과 해석과 자료 기준

원본 발표자료에는 세 백본의 비교 실험과 ResNet50 선택 내용이 있습니다. 공개 저장소에서는 **현재 포함된 소스의 구조와 구현 범위**를 결과로 소개하며, 발표자료의 수치를 배포 모델의 재현 성능으로 표기하지 않습니다.

근거 자료: 딥러닝 기말과제 발표자료 4~11쪽(전처리·모델), 12~13쪽(시스템·화면), 16쪽(팀 역할), 원본 학습·API·Streamlit 코드. 페이지는 PDF의 첫 장을 1쪽으로 셉니다.
