import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# ------------------------------------------------------------------
# 1. 모델 구조 정의 (6개 클래스로 수정됨)
# ------------------------------------------------------------------
def build_model():
    model = Sequential()
    # 1st Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Block
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Dense Block
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # ★ 수정됨: 7 -> 6
    model.add(Dense(6, activation='softmax'))
    
    return model

# ------------------------------------------------------------------
# 2. 모델 로드
# ------------------------------------------------------------------
classifier = build_model()

try:
    classifier.load_weights('model_weights.weights.h5')
    print("6개 클래스 모델 가중치 로드 성공")
except Exception as e:
    print(f"가중치 로드 실패: {e}")
    exit()

# ------------------------------------------------------------------
# 3. 설정 (Disgust 제거된 라벨)
# ------------------------------------------------------------------
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ★ 수정됨: Disgust 제거 (알파벳 순서)
emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_COLORS = {
    'Angry':    (0, 0, 255),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 255, 0),
    'Neutral':  (200, 200, 200),
    'Sad':      (255, 0, 0),
    'Surprise': (0, 255, 255)
}

H, W, C = 48, 48, 1

def preprocess_for_model(frame_bgr, box):
    x, y, w, h = box
    face_bgr = frame_bgr[y:y+h, x:x+w]
    if face_bgr.size == 0: return None
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi

def draw_label(frame, text, pos, color):
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x, y - th - baseline - 6), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def draw_legend(frame):
    x0, y0 = 10, 20
    for i, lab in enumerate(emotion_labels):
        col = EMOTION_COLORS.get(lab, (255, 255, 255))
        y = y0 + i*22
        cv2.rectangle(frame, (x0, y-12), (x0+18, y+6), col, -1)
        cv2.putText(frame, lab, (x0+26, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

# ------------------------------------------------------------------
# 4. 실행
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = preprocess_for_model(frame, (x, y, w, h))
            if roi is None: continue

            preds = classifier.predict(roi, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = emotion_labels[idx]
            conf = float(preds[idx])
            color = EMOTION_COLORS.get(label, (0,255,255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            draw_label(frame, f"{label} ({conf:.2f})", (x, y), color)

        draw_legend(frame)

    except Exception as e:
        pass

    cv2.imshow('Emotion Detector (6 Classes)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()