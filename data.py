import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool  
import message_filters

import cv2
import numpy as np
import os
import csv
import datetime
import math
from typing import List, Tuple, Any

SAVE_DIR: str = 'datasets'
IMAGE_FOLDER: str = os.path.join(SAVE_DIR, 'images')
LOG_FILE: str = os.path.join(SAVE_DIR, 'labels','labels.csv')
SYNC_SLOP: float = 0.05 

class DAVE2DataLogger(Node):
    def __init__(self):
        super().__init__('dave2_data_logger_node')
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)
            
        labels_dir = os.path.dirname(LOG_FILE)
        if not os.path.exists(labels_dir):
             os.makedirs(labels_dir)
        
        self.csv_file = open(LOG_FILE, 'a', newline='') 
        self.csv_writer = csv.writer(self.csv_file)
        
        # RNN 학습을 위해 필요한 'turn_mode' 필드를 포함합니다.
        if os.path.getsize(LOG_FILE) == 0:
            self.csv_writer.writerow([
                'image_path', 
                'turn_mode',            # 회전 여부 (True/False, 외부 토픽에서 수신)
                'linear_velocity_x',    # 선속도 (m/s)
                'angular_velocity_z'    # 각속도 (rad/s)
            ])

        # --- ROS 2 파라미터 선언 ---
        self.declare_parameter('image_topic', 'camera/color/image_raw/compressed')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel') 
        self.declare_parameter('turn_mode_topic', '/turn_mode') # <--- Turn Mode 외부 토픽 파라미터 선언

        image_topic: str = self.get_parameter('image_topic').get_parameter_value().string_value
        cmd_vel_topic: str = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        turn_mode_topic: str = self.get_parameter('turn_mode_topic').get_parameter_value().string_value # <--- 토픽 이름 가져오기

        # --- Message Filters를 이용한 동기화 구독 설정 ---
        self.image_sub = message_filters.Subscriber(self, CompressedImage, image_topic)
        self.cmd_vel_sub = message_filters.Subscriber(self, Twist, cmd_vel_topic)
        self.turn_mode_sub = message_filters.Subscriber(self, Bool, turn_mode_topic) # <--- Bool 메시지 구독자 추가

        # ApproximateTimeSynchronizer 설정: 
        # 이미지, 속도, Turn Mode (총 3개) 구독자를 포함하도록 업데이트합니다.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.cmd_vel_sub, self.turn_mode_sub], # <--- 3개 구독자 리스트
            queue_size=10, 
            slop=SYNC_SLOP,
            allow_headerless=True 
        )
        self.ts.registerCallback(self.sync_callback)

        self.save_count: int = 0
        self.get_logger().info(f"토픽 동기화 오차 허용 범위(Slop): {SYNC_SLOP}초")
        self.get_logger().info(f"Turn Mode 토픽 구독: {turn_mode_topic}")

    def sync_callback(self, img_msg: CompressedImage, cmd_vel_msg: Twist, turn_mode_msg: Bool):
        """
        CompressedImage, Twist, Bool 메시지가 동기화되어 도착하면 호출됩니다.
        """
        # cmd_vel 메시지에서 선형 속도(linear.x)와 각속도(angular.z)를 추출
        linear_x: float = cmd_vel_msg.linear.x
        angular_z: float = cmd_vel_msg.angular.z

        # --- 1. 'turn_mode' (회전 모드) 값 추출 ---
        # 외부 토픽에서 전달된 bool 값을 그대로 사용합니다.
        turn_mode: bool = turn_mode_msg.data

        # --- 2. 이미지 처리 ---
        try:
            # CompressedImage 데이터를 OpenCV 이미지로 디코딩
            np_arr: np.ndarray = np.frombuffer(img_msg.data, np.uint8)
            cv_image: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 이미지 디코딩 실패 확인 (cv_image가 None일 수 있음)
            if cv_image is None:
                self.get_logger().error("CompressedImage 디코딩 실패 (잘못된 이미지 데이터 또는 포맷)")
                return # 저장하지 않고 콜백 종료

            # --- 3. 데이터 저장 ---
            self.save_data(cv_image, turn_mode, linear_x, angular_z) 
            
        except Exception as e:
            # 이미지 디코딩 또는 저장 중 오류 발생 시 로깅
            self.get_logger().error(f"데이터 처리 또는 저장 중 오류 발생: {e}")

    def save_data(self, image_frame: np.ndarray, turn_mode: bool, linear_velocity: float, angular_velocity: float):
        """
        이미지 프레임과 제어 데이터를 저장 디렉토리에 파일로 기록합니다.
        """
        # 타임스탬프 기반 파일 이름 생성 (밀리초 단위까지 포함)
        timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        image_filename: str = f"center_{timestamp}.jpg"
        
        # 파일 경로 설정
        # 1. 상대 이미지 경로: CSV 파일에서 참조할 때 사용할 경로 (예: images/{timestamp}.jpg)
        relative_image_path: str = os.path.join(os.path.basename(IMAGE_FOLDER), image_filename)
        
        # 2. 전체 이미지 저장 경로: IMAGE_FOLDER 경로를 사용하도록 수정
        full_image_path: str = os.path.join(IMAGE_FOLDER, image_filename) # <--- 이 부분이 수정됨
        
        # 이미지 저장 (DAVE2는 전방 카메라 이미지 하나를 사용)
        cv2.imwrite(full_image_path, image_frame)
        
        # CSV 파일에 로그 기록: 'image_path', 'turn_mode', 'linear_velocity_x', 'angular_velocity_z'
        self.csv_writer.writerow([
            relative_image_path, 
            turn_mode, 
            f"{linear_velocity:.4f}", 
            f"{angular_velocity:.4f}"
        ])
        
        # 실시간 저장을 위해 버퍼를 강제로 파일에 기록 (가장 중요)
        self.csv_file.flush() 

        self.save_count += 1
        
        self.get_logger().info(
            f"[{self.save_count:05d}th] 💾 IMG: {image_filename} | Turn: {turn_mode} | "
            f"Linear X: {linear_velocity:.4f} m/s | Angular Z: {angular_velocity:.4f} rad/s"
        )

    def __del__(self):
        """
        소멸자: 노드 종료 시 열려 있는 CSV 파일을 안전하게 닫습니다.
        """
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
            self.get_logger().info("CSV 파일이 닫혔습니다.")

def main(args: List[str] = None):
    rclpy.init(args=args)
    logger = DAVE2DataLogger() 
    try:
        rclpy.spin(logger)
    except KeyboardInterrupt:
        logger.get_logger().info('노드 종료 요청 (Ctrl+C).')
    finally:
        logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
