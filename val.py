import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage 
from geometry_msgs.msg import Twist 
from std_msgs.msg import Bool 
import message_filters 
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 320, 640, 3 
FINAL_MODEL_SAVE_PATH = 'best_model.pth' 

def preprocess_image(img):
    img_resized = cv2.resize(img, (640, 320))
    img_roi = img_resized[160:320, 0:640] 
    img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    img_final = cv2.resize(img_roi, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
    return img_final

class ImprovedNeuralModel(nn.Module):
    TURN_MODE_SCALE_FACTOR = 12800.0 
    
    def __init__(self):
        super(ImprovedNeuralModel, self).__init__()
        
        CNN_OUTPUT_SIZE = 12800 
        SCALAR_INPUT_SIZE = 1 
        TOTAL_FC_INPUT_SIZE = CNN_OUTPUT_SIZE + SCALAR_INPUT_SIZE 
        
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
        
        self.output = nn.Linear(10, 1) 

    def forward(self, x_img, x_scalar):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x_img)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        
        x = self.flatten(x)
        
        scaled_scalar = x_scalar * self.TURN_MODE_SCALE_FACTOR 
        x = torch.cat((x, scaled_scalar), dim=1)
        
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.drop_fc3(F.relu(self.fc3(x)))
        
        x = self.output(x)
        return x

class DriverNode(Node):
    def __init__(self):
        super().__init__('driver_node')
        
        self.bridge = CvBridge() 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.get_logger().info(f" PyTorch 실행 장치: {self.device}")
        
        self.model = ImprovedNeuralModel()
        self.load_model()
        self.model.to(self.device)
        self.model.eval() 

        SYNC_SLOP: float = 0.05 
        
        self.image_sub = message_filters.Subscriber(self, CompressedImage, '/camera/color/image_raw/compressed')
        self.turn_mode_sub = message_filters.Subscriber(self, Bool, '/turn_mode') 
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.turn_mode_sub], 
            queue_size=10, 
            slop=SYNC_SLOP,
            allow_headerless=True 
        )
        
        self.ts.registerCallback(self.synchronized_callback)
        self.get_logger().info(" 이미지&Turn Mode 토픽 동기화 구독 시작")
        
        self.linear = 0.3
        self.publisher_ = self.create_publisher(
            Twist,
            '/cmd_vel', 
            10
        )
        self.get_logger().info("제어 토픽 발행 시작: /cmd_vel")

    def load_model(self):
        if not os.path.exists(FINAL_MODEL_SAVE_PATH):
            self.get_logger().error(f" 모델 파일이 없습니다: {FINAL_MODEL_SAVE_PATH}")
            self.get_logger().error("임의의 (훈련되지 않은) 모델로 실행됩니다.")
            return

        try:
            self.model.load_state_dict(
                torch.load(FINAL_MODEL_SAVE_PATH, map_location=self.device)
            )
            self.get_logger().info(f" 모델 가중치 로드 성공: {FINAL_MODEL_SAVE_PATH}")
        except Exception as e:
            self.get_logger().error(f" 모델 로드 중 오류 발생: {e}")


    def synchronized_callback(self, image_msg: CompressedImage, turn_mode_msg: Bool):
        """이미지 메시지와 Turn Mode 메시지가 동기화되어 수신될 때 호출됩니다."""
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                self.get_logger().error("❌ 이미지 디코딩 실패.")
                return

        except Exception as e:
            self.get_logger().error(f"이미지 디코딩/처리 오류: {e}")
            return
        
        current_turn_mode_value = 1.0 if turn_mode_msg.data else 0.0

        omega_z = self.infer_control(cv_image, current_turn_mode_value)

        twist_msg = Twist()
        twist_msg.linear.x = self.linear_x
        twist_msg.angular.z = float(omega_z)

        self.publisher_.publish(twist_msg)
        self.get_logger().info(f' 발행: V_x={self.linear_x:.4f}, Omega_z={omega_z:.4f} (Turn Mode: {current_turn_mode_value})')

    def infer_control(self, cv_image, current_turn_mode_value: float):
        preprocessed_img = preprocess_image(cv_image) 
        
        img_tensor = (preprocessed_img / 255.0) - 0.5
        img_tensor = np.transpose(img_tensor, (2, 0, 1)) 
        
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        scalar_input_tensor = torch.tensor([current_turn_mode_value], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor, scalar_input_tensor)
            prediction = outputs.cpu().numpy()[0] 

        angular_velocity_z = prediction[0]

        # Angular velocity clipping (enabled if needed)
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
