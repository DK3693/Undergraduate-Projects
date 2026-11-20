# -*- coding: utf-8 -*-
"""
CK Facial Expression Training (CK-only, stratified random split, 3-way)
- 입력: RGB 3채널 (CK+ 원본은 흑백이지만 RGB로 로드)
- 백본: VGG16 / ResNet50 / EfficientNetB0 (ImageNet 프리트레인)

데이터 구조 (원본)
- CK+: ck_root/
    - anger, contempt, disgust, fear, happy, sadness, surprise

최종 감정 라벨 (4 classes, neutral 없음):
    0: anger
    1: happy
    2: sadness
    3: nervousness  (fear + surprise)

기능:
    - subject 기반이 아닌, 클래스 비율 유지(Stratified) random train/val/test 3분할
    - class_weight 자동 계산
    - 데이터 증강은 train dataset 구성 시에만 적용
    - 각 백본별 (epochs, patience) 그리드서치 (2단계: freeze + fine-tuning)
    - 각 백본별 best 설정에 대해:
        - 전체 모델 저장 (SavedModel 형식, h5py X)
        - Loss/Accuracy 곡선 (PNG)
        - test set Confusion Matrix + Classification Report (PNG + TXT)
    - 증강이 적용된 train dataset에서 각 클래스별 예시 1장 PNG 저장
"""

import os
import glob
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report

# ======================================================
# 0. 기본 설정
# ======================================================
RESULT_DIR = "ck_only_results"
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SIZE = (96, 96)
BATCH_SIZE = 16

NUM_CLASSES = 4
NEW_CLASS_NAMES = ["anger", "happy", "sadness", "nervousness"]

AUTOTUNE = tf.data.AUTOTUNE

# 🔧 CK 경로 (네 환경에 맞게 수정)
CK_ROOT = os.environ.get("CK_ROOT", "data/ck")
CK_DIR = CK_ROOT

CK_CLASS_NAMES = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# 0: anger, 1: happy, 2: sadness, 3: nervousness
CK_MAPPING = {
    0: 0,  # anger    -> anger
    1: 0,  # contempt -> anger
    2: 0,  # disgust  -> anger
    3: 3,  # fear     -> nervousness
    4: 1,  # happy    -> happy
    5: 2,  # sadness  -> sadness
    6: 3,  # surprise -> nervousness
}

RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 🔧 그리드서치 후보 (원하면 여기만 바꾸면 됨)
EPOCH_GRID = [200]   # 총 epoch 수 (1단계+2단계 합산)
PATIENCE_GRID = [5, 10] # EarlyStopping patience

# ======================================================
# 1. CK+ stratified random 3-way split + Dataset 생성
# ======================================================

def build_stratified_split(
    ck_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=1337,
):
    """
    CK+ 전체 이미지를 대상으로,
    4개 최종 클래스 비율을 유지하면서 train/val/test로 random split.
    """
    all_paths = []
    all_new_labels = []

    # 원본 7클래스를 돌면서, 최종 4클래스로 매핑
    for orig_label, class_name in enumerate(CK_CLASS_NAMES):
        class_dir = os.path.join(ck_dir, class_name)
        pattern = os.path.join(class_dir, "*")
        file_list = glob.glob(pattern)

        for filepath in file_list:
            all_paths.append(filepath)
            all_new_labels.append(CK_MAPPING[orig_label])

    all_paths = np.array(all_paths)
    all_new_labels = np.array(all_new_labels, dtype=np.int32)

    n_total = len(all_paths)
    print(f"[INFO] Loaded total {n_total} images.")

    # 클래스별 index 모으기
    indices_per_class = {}
    for c in range(NUM_CLASSES):
        indices_per_class[c] = np.where(all_new_labels == c)[0]

    rng = np.random.default_rng(seed)

    train_idx_list = []
    val_idx_list = []
    test_idx_list = []

    for c in range(NUM_CLASSES):
        idx_c = indices_per_class[c]
        rng.shuffle(idx_c)

        n_c = len(idx_c)
        n_train_c = int(n_c * train_ratio)
        n_val_c = int(n_c * val_ratio)
        n_test_c = n_c - n_train_c - n_val_c  # 나머지

        train_idx_list.append(idx_c[:n_train_c])
        val_idx_list.append(idx_c[n_train_c:n_train_c + n_val_c])
        test_idx_list.append(idx_c[n_train_c + n_val_c:])

        print(f"[INFO] Class {c} -> total={n_c}, "
              f"train={n_train_c}, val={n_val_c}, test={n_test_c}")

    train_idx = np.concatenate(train_idx_list)
    val_idx   = np.concatenate(val_idx_list)
    test_idx  = np.concatenate(test_idx_list)

    # 각 split 내부에서 한 번 더 섞어주기 (선택 사항이지만 보기 좋게)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_paths = all_paths[train_idx].tolist()
    val_paths   = all_paths[val_idx].tolist()
    test_paths  = all_paths[test_idx].tolist()

    train_labels = all_new_labels[train_idx].tolist()
    val_labels   = all_new_labels[val_idx].tolist()
    test_labels  = all_new_labels[test_idx].tolist()

    print(f"[INFO] #images total={n_total}, "
          f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ======================================================
# 2. 데이터 증강 레이어 & Dataset 생성
# ======================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ],
    name="data_augmentation",
)

def make_dataset(paths, labels, shuffle=True, augment=False):
    """
    paths: 이미지 경로 리스트
    labels: int 라벨 리스트 (0 ~ NUM_CLASSES-1)
    augment: True이면 data_augmentation을 적용 (train 전용)
    -> (image, label_int) tf.data.Dataset 생성
    """
    paths  = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_img(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3)  # RGB
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)  # 0~255

        if augment:
            # train dataset에만 증강 적용
            img = data_augmentation(img, training=True)

        # 간단히 0~1 스케일링
        img = img / 255.0

        return img, label

    ds = ds.map(_load_img, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=RANDOM_SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


(train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
    build_stratified_split(CK_DIR)

# train에는 augmentation 적용
train_ds = make_dataset(train_paths, train_labels, shuffle=True, augment=True)
val_ds   = make_dataset(val_paths,   val_labels,   shuffle=False, augment=False)
test_ds  = make_dataset(test_paths,  test_labels,  shuffle=False, augment=False)

# class_weight 자동 계산 (train 기준)
cnt = Counter(train_labels)
total_train = len(train_labels)
class_weight = {}
for c in range(NUM_CLASSES):
    class_weight[c] = total_train / (NUM_CLASSES * cnt[c])

print("\n[INFO] Train class counts:", cnt)
print("[INFO] Class weights:", class_weight)

# ======================================================
# 3. 데이터 증강 예시 (파일 저장, augmentation 적용된 train_ds 사용)
# ======================================================

def save_augmented_examples(dataset, class_names, max_search=500):
    """
    augmentation이 이미 적용된 train_ds에서
    각 클래스별로 1장의 이미지를 뽑아서 PNG로 저장.
    (dataset은 0~1 스케일된 이미지라고 가정)
    """
    examples = {}  # class_id -> image tensor

    # dataset은 batch 단위이므로 unbatch 후 순회
    for img, label in dataset.unbatch().take(max_search):
        label_int = int(label.numpy())
        if label_int not in examples:
            examples[label_int] = img  # 이미 augmentation 적용된 상태
        if len(examples) == len(class_names):
            break

    if len(examples) < len(class_names):
        print("[WARN] 일부 클래스의 예시 이미지를 찾지 못했음.")

    fig = plt.figure(figsize=(4 * len(class_names), 4))
    for c in range(len(class_names)):
        if c not in examples:
            continue
        img = examples[c]  # 0~1 이미지 (증강 적용 완료)

        plt.subplot(1, len(class_names), c + 1)
        plt.imshow(img.numpy())
        plt.title(f"augmented: {class_names[c]}")
        plt.axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, "augmented_examples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved augmented examples to {out_path}")


print("\n[INFO] 증강 예시 이미지 저장...")
# 바로 augmentation 적용된 train_ds에서 뽑기
save_augmented_examples(train_ds, NEW_CLASS_NAMES)

# ======================================================
# 4. 모델 정의 (RGB 입력, ImageNet 프리트레인 백본)
#    -> data_augmentation, Rescaling은 이미 Dataset 단계에서 수행
# ======================================================

def build_model(backbone_name: str,
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                num_classes=NUM_CLASSES):
    """
    backbone_name: "vgg16", "resnet50", "efficientnetb0"
    - 입력: (H, W, 3) RGB, 0~1 스케일 (Dataset에서 전처리 완료)
    - 백본 + 분류 헤드
    """
    inputs = layers.Input(shape=input_shape)

    # Dataset에서 augmentation + rescaling(0~1) 완료된 상태로 들어옴
    x = inputs

    base_input_shape = (input_shape[0], input_shape[1], 3)

    backbone_lower = backbone_name.lower()
    if backbone_lower == "vgg16":
        base = VGG16(include_top=False, weights="imagenet",
                     input_shape=base_input_shape)
    elif backbone_lower == "resnet50":
        base = ResNet50(include_top=False, weights="imagenet",
                        input_shape=base_input_shape)
    elif backbone_lower == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet",
                              input_shape=base_input_shape)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # 1단계: freeze
    base.trainable = False

    features = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(features)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name=f"{backbone_name}_emotion")

    return model, base

# ======================================================
# 5. Confusion Matrix & 평가 (파일 저장)
# ======================================================

def evaluate_and_save_confusion(model, test_ds, exp_name, backbone_name):
    """
    test_ds 전체에 대해 예측 -> Confusion Matrix & Classification Report 계산,
    - Confusion Matrix 그림 PNG로 저장
    - Confusion Matrix + Classification Report 텍스트를 .txt 파일로 저장
    """
    y_true, y_pred = [], []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true, y_pred, target_names=NEW_CLASS_NAMES, digits=3
    )

    # 텍스트 저장
    txt_path = os.path.join(
        RESULT_DIR,
        f"{exp_name}_{backbone_name}_confusion_report.txt"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[Confusion Matrix] {exp_name} - {backbone_name}\n")
        f.write(str(cm) + "\n\n")
        f.write("[Classification Report]\n")
        f.write(report + "\n")

    # 그림 저장
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{exp_name} - {backbone_name} Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(NEW_CLASS_NAMES, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(NEW_CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()

    fig_path = os.path.join(
        RESULT_DIR,
        f"{exp_name}_{backbone_name}_confusion_matrix.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved confusion matrix & report for {backbone_name} to:")
    print(f"       {fig_path}")
    print(f"       {txt_path}")

# ======================================================
# 6. 단일 설정 (epochs, patience) 학습: 1단계 + 2단계(fine-tuning)
# ======================================================

def train_one_setting(
    backbone_name,
    train_ds,
    val_ds,
    epochs_total,
    patience,
    exp_name,
):
    print(f"\n=== [{backbone_name}] epochs={epochs_total}, patience={patience} ===")
    model, base = build_model(backbone_name)

    # 1) 가중치 파일(.weights.h5)로 저장하도록 경로 설정
    ckpt_path = os.path.join(
        RESULT_DIR,
        f"{exp_name}_{backbone_name}_ep{epochs_total}_pat{patience}.weights.h5"
    )

    # -----------------------
    # 1단계: backbone freeze
    # -----------------------
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs_stage1 = max(5, epochs_total // 2)

    callbacks1 = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,   # ✅ 가중치만 저장
        ),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks1,
        class_weight=class_weight,
        verbose=1,
    )

    # -----------------------
    # 2단계: fine-tuning
    # -----------------------
    remaining_epochs = epochs_total - epochs_stage1
    history2 = None

    if remaining_epochs > 0:
        print(f"[INFO] Fine-tuning {backbone_name} backbone ...")
        base.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks2 = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,  # ✅ 동일하게
            ),
        ]

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage1 + remaining_epochs,
            initial_epoch=epochs_stage1,
            callbacks=callbacks2,
            class_weight=class_weight,
            verbose=1,
        )

    # -----------------------
    # history 합치기
    # -----------------------
    history_dict = {}
    for k, v in history1.history.items():
        history_dict[k] = list(v)

    if history2 is not None:
        for k, v in history2.history.items():
            history_dict.setdefault(k, []).extend(v)

    # -----------------------
    # 2) 저장된 best 가중치를 현재 model에 로드
    # -----------------------
    model.load_weights(ckpt_path)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[VAL] {backbone_name} ep={epochs_total}, pat={patience} "
          f"-> val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    return model, history_dict, val_loss, val_acc, ckpt_path


# ======================================================
# 7. Loss/Accuracy 곡선 저장 (best run 전용)
# ======================================================

def save_loss_curves(history_dict, exp_name, backbone_name, title_prefix=""):
    # history에서 꺼내오기
    loss     = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    acc      = history_dict.get("accuracy", [])
    val_acc  = history_dict.get("val_accuracy", [])

    # 🔥 1) best epoch 인덱스 찾기 (val_loss 기준)
    if len(val_loss) > 0:
        best_epoch_idx = int(np.argmin(val_loss))  # 0-based index
    else:
        best_epoch_idx = len(loss) - 1  # 혹시 모를 안전장치

    # 🔥 2) best epoch까지로 잘라주기 (그래프에 patience 구간 안 보이도록)
    end = best_epoch_idx + 1  # 슬라이싱은 끝 인덱스가 exclusive니까 +1
    loss     = loss[:end]
    val_loss = val_loss[:end]
    acc      = acc[:end]
    val_acc  = val_acc[:end]

    epochs_range = range(1, end + 1)  # 1 ~ best_epoch

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # -------- Loss 그래프 --------
    axes[0].plot(epochs_range, loss,     label="train_loss")
    axes[0].plot(epochs_range, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix}{exp_name} - {backbone_name} Loss")
    axes[0].legend()

    # -------- Accuracy 그래프 --------
    axes[1].plot(epochs_range, acc,     label="train_acc")
    axes[1].plot(epochs_range, val_acc, label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title_prefix}{exp_name} - {backbone_name} Accuracy")  # 🔧 setTitle → set_title 수정
    axes[1].legend()

    fig.tight_layout()

    fig_path = os.path.join(
        RESULT_DIR,
        f"{exp_name}_{backbone_name}_loss_acc.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved training curves to {fig_path} (up to best epoch = {end})")


# ======================================================
# 8. 전체 실험 실행 (백본별 그리드서치)
# ======================================================

if __name__ == "__main__":
    exp_name = "ck_only_stratified_split"
    backbones = ["resnet50"]

    summary_path = os.path.join(RESULT_DIR, f"{exp_name}_grid_search_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fsum:
        fsum.write(f"Grid Search Summary for {exp_name}\n")
        fsum.write("=" * 60 + "\n\n")

        for backbone in backbones:
            print(f"\n==================\n[BACKBONE] {backbone}\n==================")

            # 각 백본별 그리드서치 결과를 저장할 리스트
            grid_results = []  # [{'epochs':..., 'patience':..., 'val_loss':..., 'val_acc':...}, ...]

            best_val_loss = float("inf")
            # (epochs, patience, val_loss, val_acc, ckpt_path, history_dict, best_model)
            best_info = None

            for epochs_total in EPOCH_GRID:
                for patience in PATIENCE_GRID:
                    best_model_tmp, history_dict_tmp, val_loss, val_acc, ckpt_path = train_one_setting(
                        backbone_name=backbone,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        epochs_total=epochs_total,
                        patience=patience,
                        exp_name=exp_name,
                    )

                    # 현재 설정 결과 기록
                    grid_results.append({
                        "epochs": epochs_total,
                        "patience": patience,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "ckpt_path": ckpt_path,
                        "history": history_dict_tmp,
                        "model": best_model_tmp,
                    })

                    # best 갱신
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_info = (
                            epochs_total,
                            patience,
                            val_loss,
                            val_acc,
                            ckpt_path,
                            history_dict_tmp,
                            best_model_tmp,
                        )

            # --------- 그리드서치 결과 출력 (콘솔) ---------
            print(f"\n[GRID SEARCH RESULTS] backbone = {backbone}")
            for r in grid_results:
                print(
                    f"  ep={r['epochs']:>4}, pat={r['patience']:>3} "
                    f"-> val_loss={r['val_loss']:.4f}, val_acc={r['val_acc']:.4f}"
                )

            # 백본별 best 설정으로 최종 평가
            best_epochs, best_patience, best_vloss, best_vacc, best_ckpt_path, best_history, best_model = best_info

            print(
                f"\n[BEST] {backbone} -> "
                f"epochs={best_epochs}, patience={best_patience}, "
                f"val_loss={best_vloss:.4f}, val_acc={best_vacc:.4f}"
            )

            # --------- summary 파일에도 기록 ---------
            fsum.write(f"[BACKBONE] {backbone}\n")
            fsum.write("  Grid search results:\n")
            for r in grid_results:
                fsum.write(
                    f"    ep={r['epochs']:>4}, pat={r['patience']:>3} "
                    f"-> val_loss={r['val_loss']:.4f}, val_acc={r['val_acc']:.4f}\n"
                )

            fsum.write("  BEST CONFIG:\n")
            fsum.write(f"    best_epochs   = {best_epochs}\n")
            fsum.write(f"    best_patience = {best_patience}\n")
            fsum.write(f"    best_val_loss = {best_vloss:.4f}\n")
            fsum.write(f"    best_val_acc  = {best_vacc:.4f}\n\n")

            # ✅ 평가 전에 compile
            best_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            best_model.load_weights(best_ckpt_path)

            # test 평가 + confusion matrix / report 저장
            test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
            print(f"\n[TEST] {exp_name} - {backbone}: "
                  f"loss={test_loss:.4f}, acc={test_acc:.4f}")

            fsum.write(f"  test_loss     = {test_loss:.4f}\n")
            fsum.write(f"  test_acc      = {test_acc:.4f}\n\n")
            fsum.write("-" * 60 + "\n\n")

            evaluate_and_save_confusion(best_model, test_ds, exp_name, backbone)

            # best run training curves 저장
            save_loss_curves(best_history, exp_name, backbone, title_prefix="[CK-only Emotion] ")

        print(f"\n[INFO] Grid search summary saved to {summary_path}")
