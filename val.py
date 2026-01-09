import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ROS 2 메시지 타입
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

# --- 1. 설정값 (훈련 코드와 일치해야 함) ---
MODEL_PATH = 'best_dave2_cnn_pure_regression_model.pth' # 🚨 학습된 모델 파일 경로
IMAGE_CHANNELS = 3

# 이미지 크기 설정 (학습 코드와 일치)
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
TARGET_IMG_HEIGHT = 480 
TARGET_IMG_WIDTH = 640

# PyTorch 디바이스 설정
DEVICE = torch.device('cpu') 

# ==============================================================================
# --- 2. PyTorch 모델 정의 (Pure DAVE2 CNN Regression) ---
# ==============================================================================
DUMMY_INPUT_SIZE = (1, IMAGE_CHANNELS, TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH)

class Dave2CNNRegressionModel(nn.Module):
    
    def __init__(self):
        super(Dave2CNNRegressionModel, self).__init__()

        self.cnn_base = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 24, kernel_size=5, stride=2, padding=2), nn.ELU(), 
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2), nn.ELU(),   
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2), nn.ELU(),   
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0), nn.ELU(),   
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ELU(),   
            nn.Flatten()
        )
        
        try:
            with torch.no_grad():
                # feature_dim 계산 (훈련 코드와 동일해야 함)
                dummy_output = self.cnn_base(torch.zeros(*DUMMY_INPUT_SIZE))
                self.feature_dim = dummy_output.size(1) 
        except Exception:
            self.feature_dim = 272384 # Fallback 값
            
        # 최종 Dense Layer: CNN 특징만 입력, 출력 2
        self.output_dense = nn.Sequential(
            nn.Linear(self.feature_dim, 100), nn.ELU(),
            nn.Dropout(0.5), 
            nn.Linear(100, 50), nn.ELU(),
            nn.Dropout(0.5), 
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 2) # 단일 출력 (w, v)
        )

    # 🚨 forward 함수의 입력은 이미지 텐서만 받음
    def forward(self, x): 
        # x shape: (B, C, H, W)
        cnn_features = self.cnn_base(x)
        final_output = self.output_dense(cnn_features)
        
        return final_output 

# ==============================================================================
# --- 3. 전처리 함수 (훈련 코드와 동일) ---
# ==============================================================================

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """훈련 시 사용한 전처리 과정을 적용합니다."""
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_normalized = (img_resized.astype(np.float32) / 255.0) - 0.5
    img_final = np.transpose(img_normalized, (2, 0, 1))
    
    return img_final

# ==============================================================================
# --- 4. ROS 2 컨트롤러 노드 ---
# ==============================================================================

class Dave2CNNController(Node):
    def __init__(self):
        super().__init__('dave2_cnn_controller') 
        
        self.device = DEVICE 
        self.twist_msg = Twist()
        
        # 1. 모델 로드
        self.load_model()
        
        # 2. ROS 2 구독/발행 설정
        # 🚨 Bool 토픽 구독 없음
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed', self.image_callback, 1) 
            
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        
        self.get_logger().info("🚀 Pure DAVE2 CNN Controller 시작. 모든 제한 해제.")

    def load_model(self):
        """모델 로드 및 오류 처리"""
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"❌ 오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
            raise FileNotFoundError("모델 파일 없음.")
        
        try:
            self.model = Dave2CNNRegressionModel().to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')) 
            self.model.eval()
            self.get_logger().info("✅ 모델 로드 완료. 추론 준비 완료.")
        except Exception as e:
            self.get_logger().error(f"❌ 모델 로드 오류: {e}")
            self.get_logger().error("❗ 모델 구조와 저장된 가중치 파일이 일치하는지 확인하십시오.")
            raise RuntimeError("모델 로드 실패")
            
    def image_callback(self, data: CompressedImage):
        """ 카메라 이미지 메시지를 수신하여 추론 수행 """
        
        try:
            # 1. 이미지 디코딩 및 전처리
            np_arr = np.frombuffer(data.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: return

            img = cv2.resize(img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
            processed_img = preprocess_image(img) # (C, H, W)
            
            # 2. 추론 데이터 준비: (B=1, C, H, W)
            img_tensor = torch.from_numpy(processed_img).unsqueeze(0).to(self.device) 
            
            # 3. 모델 추론
            with torch.no_grad():
                self.model.eval()
                # 🚨 이미지 텐서만 전달
                prediction_output = self.model(img_tensor) 
                
            # 4. 최종 예측: (w, v)
            output = prediction_output[0].cpu().numpy()
            
            raw_angular = output[0] # omega_z
            raw_linear = output[1]  # linear_x
            
            # 5. 제어 메시지 생성 및 발행
            cmd_vel_msg = Twist()
            
            # 🚨 어떠한 제한(클리핑) 없이 모델 출력 값을 그대로 사용
            cmd_vel_msg.linear.x = float(raw_linear) 
            cmd_vel_msg.angular.z = float(raw_angular)
            
            self.vel_pub.publish(cmd_vel_msg)
            
            self.get_logger().info(
                f"✅ Pub: $\omega_z$={cmd_vel_msg.angular.z:.4f}, $v_x$={cmd_vel_msg.linear.x:.4f} (Raw Output)", 
                throttle_duration_sec=0.05
            )

        except Exception as e:
            self.get_logger().error(f"❌ 추론/제어 오류 발생: {e}")
            self.stop_robot()

    def stop_robot(self):
        """ 로봇을 정지시키는 함수 """
        self.twist_msg.linear.x = 0.0
        self.twist_msg.angular.z = 0.0
        if hasattr(self, 'vel_pub'):
            self.vel_pub.publish(self.twist_msg)


def main(args=None):
    rclpy.init(args=args)
    controller = None
    try:
        controller = Dave2CNNController()
        rclpy.spin(controller)
    except (KeyboardInterrupt, SystemExit, FileNotFoundError, RuntimeError) as e:
        if controller:
            controller.get_logger().info(f'노드 종료: {type(e).__name__} 발생.')
    except Exception as e:
        print(f"최상위 오류: {e}")
    finally:
        if controller:
            controller.stop_robot()
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()