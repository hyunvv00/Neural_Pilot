# Neuron Pilot - ROS2 Autonomous Driving

DAVE-2 스타일의 Behavioral Cloning으로 카메라 이미지 → 스티어링 각도(angular velocity)를 예측하는 자율주행 모델입니다. 
ROS2 카메라 토픽에서 실시간 이미지를 받아 PyTorch CNN으로 스티어링 명령을 생성하고, turn mode 입력을 결합해 더 정교한 제어를 구현했습니다.

> 핵심: NVIDIA DAVE-2와 달리 turn mode(직진/회전) 스칼라 입력을 추가해 상황별 스티어링을 강화했습니다.

---

## 디렉토리 구조

```text
├── data_logger.py                 # 데이터 수집: 카메라 + cmd_vel + turn_mode 동시 로깅(CSV + JPG)
├── train.py                       # 모델 학습: PyTorch CNN 학습 + 데이터 증강 + 클래스 밸런싱
└── val.py                         # 실시간 추론: ROS2 노드로 카메라 입력 → 스티어링 출력
```

---

## 모델 구조
- ROI Crop: 이미지 상단 160~320행(도로 영역)만 사용
- 데이터 증강: 좌우반전/밝기/그림자/노이즈
- 클래스 밸런싱: 직진/약회전/급회전 비율 맞춤
- Weighted Loss: MSE(70%) + MAE(30%) + 큰 오차 페널티
```python
입력: 이미지(320x640x3) + Turn Mode(스칼라 0/1)
    ↓
Conv1(24ch,5x5,s2) → BN → ReLU → MaxPool → Dropout(0.1)
Conv2(36ch,5x5,s2) → BN → ReLU → MaxPool → Dropout(0.1)  
Conv3(48ch,5x5,s2) → BN → ReLU → Dropout(0.2)
Conv4(64ch,3x3,s1) → BN → ReLU → Dropout(0.2)
Conv5(64ch,3x3,s1) → BN → ReLU → Dropout(0.3)
    ↓ (Flatten: 12800)
FC1(100) → BN → ReLU → Dropout(0.4)
FC2(50) → BN → ReLU → Dropout(0.3)
FC3(10) → Dropout(0.2)
    ↓
출력: Angular Velocity (ωz, rad/s)
```

---

## 데이터 수집 (data_logger.py)​
ROS2 3개 토픽 동시 동기화:

```text
CompressedImage (카메라)
Twist (cmd_vel: linear.x, angular.z)  
Bool (turn_mode: 직진=0, 회전=1)
```

출력:
```text
datasets/
├── images/        # center_[timestamp].jpg
└── labels/labels.csv  # imagepath,turn_mode,angular_velocity_z
동기화: ApproximateTimeSynchronizer (slop=0.05s)
```
​
---

## 모델 학습 (train.py)
​
전처리 파이프라인
```text
원본 → Resize(640x320) → ROI(도로 영역) → RGB 변환 → Resize(IMAGE_WIDTHxIMAGE_HEIGHT)
```

학습 설정
- Batch Size: 512
- Optimizer: Adam(lr=2e-5, β=(0.9,0.999))
- Scheduler: ReduceLROnPlateau(patience=5)
- Early Stopping: patience=50
- Checkpoint: checkpoint.pth / bestmodel.pth

​
클래스 밸런싱
```text
직진(|ωz|<0.1): 1배
약회전(0.1≤|ωz|<0.3): 2배 오버샘플링
급회전(|ωz|≥0.3): 3배 오버샘플링
```

---

## 실시간 추론 (val.py)
​
ROS2 Driver Node:

```text
카메라 토픽 수신 → 전처리 → 모델 추론 → Twist(linear.x=0.3, angular.z=예측값) 발행
```

---

## 빠른 시작 

### 1. 환경 설정
```bash
sudo apt install ros-humble-cv-bridge ros-humble-message-filters
pip3 install torch torchvision opencv-python scikit-learn pandas numpy
```

### 2. 데이터 수집
```bash
python3 data_logger
```
출력:
```text
datasets/
├── images/        # center_[timestamp].jpg
└── labels/labels.csv  # imagepath,turn_mode,angular_velocity_z
```
### 3. 모델 학습
```bash
python3 train.py
```

### 4. 실시간 추론
```bash
python3 val.py
```

---
