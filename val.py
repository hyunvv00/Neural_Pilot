# driver_node.py
# PyTorch DAVE-2 기반 자율주행 ROS 2 노드 (W640 + Turn_Mode 입력)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage 
from geometry_msgs.msg import Twist 
from std_msgs.msg import Bool # <--- Turn Mode 데이터를 위한 Bool 메시지 임포트
import message_filters # <--- 동기화를 위한 message_filters 임포트
from cv_bridge import CvBridge # <--- 이미지 메시지 처리를 위한 CvBridge 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

# --- 1. 전역 설정 및 하이퍼파라미터 (훈련 파일과 동일) ---

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 320, 640, 3 

# 🚨 훈련 스크립트에서 저장된 파일 이름으로 변경하세요.
FINAL_MODEL_SAVE_PATH = 'best_model.pth' 

# --- 2. 모델 전처리 함수 (훈련 파일과 동일) ---

def preprocess_image(img):
    """ 
    [W640 적용] 이미지의 아래쪽 절반 (160:320)과 전체 넓이 (0:640)를 사용합니다.
    """
    img_resized = cv2.resize(img, (640, 320))
    # ROI: 아래쪽 절반 160:320
    img_roi = img_resized[160:320, 0:640] 
    img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    # 최종 DAVE-2 입력 크기 (여기서는 320x640)로 리사이즈
    img_final = cv2.resize(img_roi, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
    return img_final

# --- 3. PyTorch 모델 정의 (훈련 코드와 완벽히 일치) ---

class ImprovedDave2Model(nn.Module):
    """
    개선된 DAVE-2 PyTorch 모델 (W640 + Turn_Mode 입력)
    """
    # 🚨 turn_mode의 영향력을 강화하기 위한 스케일링 팩터 (훈련 시와 동일해야 함!)
    # 훈련 시 12800.0을 사용했다면 그대로 유지
    TURN_MODE_SCALE_FACTOR = 12800.0 
    
    def __init__(self):
        super(ImprovedDave2Model, self).__init__()
        
        CNN_OUTPUT_SIZE = 12800 
        SCALAR_INPUT_SIZE = 1 
        TOTAL_FC_INPUT_SIZE = CNN_OUTPUT_SIZE + SCALAR_INPUT_SIZE 
        
        # CNN 레이어 정의 (훈련 시와 동일)
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 24, kernel_size=5, stride=2, padding=2) 
        self.bn1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(36)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.drop3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.drop5 = nn.Dropout(0.3)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(TOTAL_FC_INPUT_SIZE, 100) 
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.drop_fc1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(100, 50)
        self.bn_fc2 = nn.BatchNorm1d(50)
        self.drop_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(50, 10)
        self.drop_fc3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(10, 1) # 출력 1개: [omega_z]

    def forward(self, x_img, x_scalar):
        # 1. CNN 처리
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x_img)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        
        x = self.flatten(x)
        
        # 🚨 2. CNN 출력과 스칼라 입력 결합 (스케일링 적용)
        scaled_scalar = x_scalar * self.TURN_MODE_SCALE_FACTOR 
        x = torch.cat((x, scaled_scalar), dim=1)
        
        # 3. FC 처리
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.drop_fc3(F.relu(self.fc3(x)))
        
        x = self.output(x)
        return x

# --- 4. ROS 2 노드 클래스 정의 (수정됨: 동기화 및 듀얼 구독) ---

class DriverNode(Node):
    def __init__(self):
        super().__init__('driver_node')
        
        self.bridge = CvBridge() # CvBridge 인스턴스 초기화
        
        # 1. 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"✅ PyTorch 실행 장치: {self.device}")
        
        # 2. PyTorch 모델 로드
        self.model = ImprovedDave2Model()
        self.load_model()
        self.model.to(self.device)
        self.model.eval() 

        # 3. ROS 2 구독 설정 및 동기화 (ApproximateTimeSynchronizer 사용)
        SYNC_SLOP: float = 0.05 
        
        # 🚨 구독자 정의
        self.image_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        self.turn_mode_sub = message_filters.Subscriber(self, Bool, '/turn_mode') 
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.turn_mode_sub], # 🚨 두 개의 구독자 리스트
            queue_size=10, 
            slop=SYNC_SLOP,
            allow_headerless=True 
        )
        
        # 🚨 동기화 콜백 등록
        self.ts.registerCallback(self.synchronized_callback)
        self.get_logger().info("✅ 이미지/Turn Mode 토픽 동기화 구독 시작")
        
        # 5. ROS 2 발행 설정 (Twist 메시지)
        self.publisher_ = self.create_publisher(
            Twist,
            '/cmd_vel', 
            10
        )
        self.get_logger().info("✅ 제어 토픽 발행 시작: /cmd_vel")

    def load_model(self):
        """훈련된 모델 가중치를 로드합니다."""
        if not os.path.exists(FINAL_MODEL_SAVE_PATH):
            self.get_logger().error(f"❌ 모델 파일이 없습니다: {FINAL_MODEL_SAVE_PATH}")
            self.get_logger().error("⚠️ 임의의 (훈련되지 않은) 모델로 실행됩니다.")
            return

        try:
            self.model.load_state_dict(
                torch.load(FINAL_MODEL_SAVE_PATH, map_location=self.device)
            )
            self.get_logger().info(f"✅ 모델 가중치 로드 성공: {FINAL_MODEL_SAVE_PATH}")
        except Exception as e:
            self.get_logger().error(f"❌ 모델 로드 중 오류 발생: {e}")


    def synchronized_callback(self, image_msg: CompressedImage, turn_mode_msg: Bool):
        """이미지 메시지와 Turn Mode 메시지가 동기화되어 수신될 때 호출됩니다."""
        try:
            # 1. CompressedImage -> OpenCV BGR Image (CvBridge 사용)
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                self.get_logger().error("❌ 이미지 디코딩 실패.")
                return

        except Exception as e:
            self.get_logger().error(f"이미지 디코딩/처리 오류: {e}")
            return
        
        # 2. Turn Mode 값 추출 및 변환 (Bool -> float)
        # Bool 메시지의 data 필드는 True/False를 가지며, 이를 1.0 또는 0.0으로 변환
        current_turn_mode_value = 1.0 if turn_mode_msg.data else 0.0

        # 3. PyTorch 추론 및 제어 값 획득
        omega_z = self.infer_control(cv_image, current_turn_mode_value)

        # 4. 제어 명령(Twist 메시지) 생성 및 발행
        twist_msg = Twist()
        twist_msg.linear.x = 0.3 # 고정 선속도
        twist_msg.angular.z = float(omega_z)

        self.publisher_.publish(twist_msg)
        self.get_logger().info(f'📢 발행: V_x={0.3:.4f}, Omega_z={omega_z:.4f} (Turn Mode: {current_turn_mode_value})')

    def infer_control(self, cv_image, current_turn_mode_value: float):
        """ OpenCV 이미지를 입력받아 전처리하고 모델 추론을 수행합니다. """
        
        # 1. 전처리 (W640 ROI 및 최종 크기로 조정)
        preprocessed_img = preprocess_image(cv_image) 

        # 2. PyTorch Tensor로 변환 및 정규화
        img_tensor = (preprocessed_img / 255.0) - 0.5
        img_tensor = np.transpose(img_tensor, (2, 0, 1)) 
        
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 🚨 스칼라 입력 텐서 준비: [current_turn_mode_value]
        # Bool 값이 True(1.0) 또는 False(0.0)로 변환된 값이 사용됨
        scalar_input_tensor = torch.tensor([current_turn_mode_value], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. 추론 (Inference)
        with torch.no_grad():
            # 🚨 이미지와 스칼라 입력을 함께 모델에 전달
            outputs = self.model(img_tensor, scalar_input_tensor)
            prediction = outputs.cpu().numpy()[0] 

        # 4. 결과 반환
        angular_velocity_z = prediction[0]

        # 🚨 각속도 클리핑 (필요시 활성화)
        # angular_velocity_z = np.clip(angular_velocity_z, -1.0, 1.0)

        return angular_velocity_z


def main(args=None):
    rclpy.init(args=args)
    driver_node = DriverNode()
    
    try:
        rclpy.spin(driver_node)
    except KeyboardInterrupt:
        pass
        
    driver_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()