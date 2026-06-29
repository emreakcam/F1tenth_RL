import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from stable_baselines3 import PPO
import os

class F110InferenceNode(Node):
    def __init__(self):
        super().__init__('f110_inference_node')

        # 1. Ayarlar
        # Model dosyasının tam yolunu buraya yaz (.zip koymana gerek yok)
        self.model_path = "f1tenth_final_3million" 
        
        # 2. İletişim (Sub/Pub)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 1)

        # 3. Model Yükleme
        if os.path.exists(self.model_path + ".zip"):
            self.get_logger().info(f"Model yükleniyor: {self.model_path}")
            # Jetson üzerinde CUDA varsa device="cuda", yoksa "cpu" yapabilirsin
            self.model = PPO.load(self.model_path, device="cpu")
        else:
            self.get_logger().error(f"MODEL BULUNAMADI: {self.model_path}.zip")
            exit()

    def scan_callback(self, msg):
        # LiDAR verisini temizle ve modele uygun hale getir
        scan_data = np.array(msg.ranges)
        scan_data[np.isinf(scan_data)] = 30.0
        scan_data[np.isnan(scan_data)] = 0.0

        # Modelden tahmin al (Inference)
        # observation formatı: [1080,]
        action, _states = self.model.predict(scan_data, deterministic=True)

        # Araca komutu gönder
        self.publish_drive(action)

    def publish_drive(self, action):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        
        # Action[0]: steering_angle, Action[1]: speed
        drive_msg.drive.steering_angle = float(action[0])
        drive_msg.drive.speed = float(action[1])
        
        self.pub_drive.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = F110InferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Durunca aracı durdur
        stop_msg = AckermannDriveStamped()
        node.pub_drive.publish(stop_msg)
        node.get_logger().info("Durduruluyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()