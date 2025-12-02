import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D

# --------------------------------------------------------------------------
# 1. 데이터 준비 (평가용)
# --------------------------------------------------------------------------
test_dir = "./test" # 검증 데이터 경로

# [중요] ./test 폴더 안에 'disgust' 폴더가 삭제되어 있어야 합니다.
# 삭제되었다면 자동으로 6개 클래스만 인식합니다.
eval_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

validation_generator = eval_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical",
    subset = "validation",
    shuffle = False  # <--- 매우 중요: 순서를 고정해야 함
)

# --------------------------------------------------------------------------
# 2. 모델 구조 재정의 및 가중치 로드
# --------------------------------------------------------------------------
def create_model():
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
    
    # ★ [수정됨] 7 -> 6 (학습 코드와 동일하게 맞춰야 함)
    model.add(Dense(6, activation='softmax'))
    
    return model

model = create_model()

# 저장된 가중치 불러오기 (반드시 6개 클래스로 새로 학습한 가중치여야 함)
weights_path = 'model_weights.weights.h5'
try:
    model.load_weights(weights_path)
    print(f"'{weights_path}' 가중치를 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"가중치 로드 실패: {e}")
    print("Shape mismatch 에러라면 Dense 층의 숫자가 맞는지 확인하세요.")
    exit()

# --------------------------------------------------------------------------
# 3. 예측 및 Confusion Matrix 생성
# --------------------------------------------------------------------------
print("예측을 수행하는 중입니다...")
# 전체 검증 데이터에 대해 예측
validation_pred_probs = model.predict(validation_generator)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

# 실제 라벨 가져오기 (shuffle=False이므로 순서가 일치함)
validation_labels = validation_generator.classes

# 클래스 이름 가져오기
class_names = list(validation_generator.class_indices.keys())
print(f"감지된 클래스: {class_names}") # ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'] 인지 확인

# Confusion Matrix 계산
confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (6 Classes)')
plt.show()

# --------------------------------------------------------------------------
# 4. 샘플 이미지 확인
# --------------------------------------------------------------------------
val_iter = iter(validation_generator)
images, labels = next(val_iter)

preds_probs = model.predict(images)
preds_indices = np.argmax(preds_probs, axis=1)
true_indices = np.argmax(labels, axis=1)

plt.figure(figsize=(10, 10))
for i in range(min(9, len(images))):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].reshape(48, 48), cmap='gray')
    
    true_cls = class_names[true_indices[i]]
    pred_cls = class_names[preds_indices[i]]
    
    col = 'green' if true_cls == pred_cls else 'red'
    
    plt.title(f"True: {true_cls}\nPred: {pred_cls}", color=col)
    plt.axis("off")
plt.show()