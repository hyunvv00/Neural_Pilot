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

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger_node')
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)
            
        labels_dir = os.path.dirname(LOG_FILE)
        if not os.path.exists(labels_dir):
             os.makedirs(labels_dir)
        
        self.csv_file = open(LOG_FILE, 'a', newline='') 
        self.csv_writer = csv.writer(self.csv_file)
        
        if os.path.getsize(LOG_FILE) == 0:
            self.csv_writer.writerow([
                'image_path', 
                'turn_mode',           
                'linear_velocity_x',   
                'angular_velocity_z'   
            ])

        self.declare_parameter('image_topic', '/camera/color/image_raw/compressd')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel') 
        self.declare_parameter('turn_mode_topic', '/turn_mode')

        image_topic: str = self.get_parameter('image_topic').get_parameter_value().string_value
        cmd_vel_topic: str = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        turn_mode_topic: str = self.get_parameter('turn_mode_topic').get_parameter_value().string_value

        self.image_sub = message_filters.Subscriber(self, CompressedImage, image_topic)
        self.cmd_vel_sub = message_filters.Subscriber(self, Twist, cmd_vel_topic)
        self.turn_mode_sub = message_filters.Subscriber(self, Bool, turn_mode_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.cmd_vel_sub, self.turn_mode_sub],
            queue_size=10, 
            slop=SYNC_SLOP,
            allow_headerless=True 
        )
        self.ts.registerCallback(self.sync_callback)

        self.save_count: int = 0
        # self.get_logger().info(f"토픽 동기화 오차 허용 범위(Slop): {SYNC_SLOP}초")
        # self.get_logger().info(f"Turn Mode 토픽 구독: {turn_mode_topic}")

    def sync_callback(self, img_msg: CompressedImage, cmd_vel_msg: Twist, turn_mode_msg: Bool):
        linear_x: float = cmd_vel_msg.linear.x
        angular_z: float = cmd_vel_msg.angular.z
        turn_mode: bool = turn_mode_msg.data
        try:
            np_arr: np.ndarray = np.frombuffer(img_msg.data, np.uint8)
            cv_image: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error("CompressedImage 디코딩 실패 (잘못된 이미지 데이터 또는 포맷)")
                return 

            self.save_data(cv_image, turn_mode, linear_x, angular_z) 
            
        except Exception as e:
            self.get_logger().error(f"데이터 처리 또는 저장 중 오류 발생: {e}")

    def save_data(self, image_frame: np.ndarray, turn_mode: bool, linear_velocity: float, angular_velocity: float):
        timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        image_filename: str = f"center_{timestamp}.jpg"
        relative_image_path: str = os.path.join(os.path.basename(IMAGE_FOLDER), image_filename)
        full_image_path: str = os.path.join(IMAGE_FOLDER, image_filename) 
        cv2.imwrite(full_image_path, image_frame)

        # CSV file: 'image_path', 'turn_mode', 'linear_velocity_x', 'angular_velocity_z'
        self.csv_writer.writerow([
            relative_image_path, 
            turn_mode, 
            f"{linear_velocity:.4f}", 
            f"{angular_velocity:.4f}"
        ])
        
        self.csv_file.flush() 

        self.save_count += 1
        
        self.get_logger().info(
            f"[{self.save_count:05d}th] IMG: {image_filename} | Turn: {turn_mode} | "
            f"Linear X: {linear_velocity:.4f} m/s | Angular Z: {angular_velocity:.4f} rad/s"
        )

    def __del__(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
            self.get_logger().info("CSV 파일이 닫혔습니다.")

def main(args: List[str] = None):
    rclpy.init(args=args)
    logger = DataLogger() 
    try:
        rclpy.spin(logger)
    except KeyboardInterrupt:
        logger.get_logger().info('노드 종료 요청 (Ctrl+C).')
    finally:
        logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
