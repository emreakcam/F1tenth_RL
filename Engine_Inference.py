import rclpy
from rclpy.node import Node
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class F110TensorRTInference(Node):
    def __init__(self):
        super().__init__('f110_trt_inference')
        
        # TensorRT Ayarları
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        
        # Engine dosyasını yükle
        with open("f1tenth_model.engine", "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Bellek Rezervasyonu (Host and Device)
        self.h_input = cuda.pagelocked_empty(1080, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(2, dtype=np.float32) # [steering, speed]
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()

        # ROS İletişimi
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 1)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.get_logger().info("TensorRT Engine başarıyla yüklendi ve çalışıyor!")

    def scan_callback(self, msg):
        # Veri Ön İşleme
        scan_data = np.array(msg.ranges, dtype=np.float32)
        scan_data[np.isinf(scan_data)] = 30.0
        scan_data[np.isnan(scan_data)] = 0.0
        
        # Giriş verisini kopyala
        np.copyto(self.h_input, scan_data)
        
        # GPU'ya transfer ve Çıkarım (Inference)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        # Aksiyonu Yayınla
        self.publish_drive(self.h_output)

    def publish_drive(self, action):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = float(action[0])
        msg.drive.speed = float(action[1])
        self.pub_drive.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = F110TensorRTInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()