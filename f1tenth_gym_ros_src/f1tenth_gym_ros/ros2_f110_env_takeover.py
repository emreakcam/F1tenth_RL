import gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from stable_baselines3 import PPO
import time
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from visualization_msgs.msg import Marker
from scipy.interpolate import CubicSpline
import os
import random # Rastgele seçim için eklendi

class F110ROSEnv(gym.Env):
    def __init__(self):
        super(F110ROSEnv, self).__init__()

        if not rclpy.ok():
            rclpy.init()
        self.node = Node("f110_rl_env")

        # Değişkenler
        self.scan = None
        self.odom = None
        self.opp_odom = None
        self.target = np.array([0.0, -1.0])
        self.current_wp_idx = 0 
        
        # Bitiş Çizgisi Tanımları
        self.finish_A = np.array([-2.0, -1.0])
        self.finish_B = np.array([-0.66, -2.1])
        self.finish_points = np.linspace(self.finish_A, self.finish_B, 50)
        self.line_threshold = 0.4
        self.second_lap_treshold = 2.0
        self.second_lap_check = False
        self.lap_count = 0
        
        # 1. Waypoint Interpolation (Pürüzsüzleştirme)
        base_waypoints = np.array([
            [3.0, -2.0], [4.0, -3.0], [4.3, -4.05], [4.04, -5.0], 
            [3.5, -5.8], [2.6, -5.80], [1.66, -5.8], [1.55, -5.6], [1.3, -5.4],
            [1.0, -5.0], [-0.37, -3.44], [-1.25, -3.0], [-1.6, -2.5],
            [-0.63, -1.16], [0.1, -0.9], [1.45, -1.03],
        ])
        base_waypoints = np.vstack([base_waypoints, base_waypoints[0]])
        t = np.arange(len(base_waypoints))
        interp_t = np.linspace(0, len(base_waypoints) - 1, 200)
        cs_x = CubicSpline(t, base_waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, base_waypoints[:, 1], bc_type='periodic')
        self.waypoints = np.stack([cs_x(interp_t), cs_y(interp_t)], axis=1)

        self.lookahead_distance = 1.0 

        self.create_communications()

        # RL Alanları: (3, 1080) Frame Stacking
        self.action_space = gym.spaces.Box(
            low=np.array([-0.4, 1.0]), high=np.array([0.4, 4.0]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=30.0, shape=(3, 1080), dtype=np.float32
        )
        self.scan_buffer = np.zeros((3, 1080), dtype=np.float32)

    def create_communications(self):
        self.node.create_subscription(LaserScan, "/scan", self.scan_cb, 1)
        self.node.create_subscription(Odometry, "/ego_racecar/odom", self.odom_cb, 1)
        self.node.create_subscription(Odometry, "/opp_racecar/odom", self.opp_odom_cb, 1)
        self.pub_drive = self.node.create_publisher(AckermannDriveStamped, "/drive", 1)
        self.pub_opp_drive = self.node.create_publisher(AckermannDriveStamped, "/opp_drive", 1)
        self.pub_init_pose = self.node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.pub_opp_init_pose = self.node.create_publisher(PoseStamped, "/goal_pose", 10)
        self.marker_pub = self.node.create_publisher(Marker, "/visualization_marker", 10)

    def scan_cb(self, msg):
        data = np.array(msg.ranges)
        data[np.isinf(data)] = 30.0
        data[np.isnan(data)] = 0.0
        self.scan = data

    def odom_cb(self, msg): self.odom = msg
    def opp_odom_cb(self, msg): self.opp_odom = msg

    def reset(self):
        print("\n--- Reset: Araçlar Belirlenen Noktalar Dışında Rastgele Atanıyor ---", flush=True)
        self.scan, self.odom, self.opp_odom = None, None, None
        
        # 1. Hariç tutulacak noktalar (Blacklist)
        blacklist = np.array([[-0.63, -1.16], [0.1, -0.9], [1.45, -1.03]])
        
        # Uygun waypoint'lerin indekslerini bul
        valid_indices = []
        for i, wp in enumerate(self.waypoints):
            # Eğer mevcut waypoint blacklist'teki noktalardan herhangi birine çok yakınsa (0.1m) alma
            is_blacklisted = any(np.linalg.norm(wp - bl) < 0.3 for bl in blacklist)
            if not is_blacklisted:
                valid_indices.append(i)
        
        # Geçerli indekslerden rastgele birini seç
        random_idx = random.choice(valid_indices)
        self.current_wp_idx = random_idx
        
        spawn_pos = self.waypoints[random_idx]
        # Botun yönünü (yaw) bir sonraki waypoint'e bakacak şekilde ayarla
        next_idx = (random_idx + 1) % len(self.waypoints)
        diff = self.waypoints[next_idx] - spawn_pos
        yaw = np.arctan2(diff[1], diff[0])
        
        opp_p = PoseStamped()
        opp_p.header.frame_id = "map"
        opp_p.pose.position.x = float(spawn_pos[0])
        opp_p.pose.position.y = float(spawn_pos[1])
        opp_p.pose.orientation.z = np.sin(yaw / 2.0)
        opp_p.pose.orientation.w = np.cos(yaw / 2.0)
        self.pub_opp_init_pose.publish(opp_p)
        
        time.sleep(0.1)

        # 2. Ego Işınla (Ego sabit noktada başlasın veya bunu da randomize edebilirsiniz)
        ego_p = PoseWithCovarianceStamped()
        ego_p.header.frame_id = "map"
        ego_p.pose.pose.position.x, ego_p.pose.pose.position.y = 0.0, -1.0
        ego_p.pose.pose.orientation.w = 1.0
        self.pub_init_pose.publish(ego_p)

        stop = AckermannDriveStamped()
        self.pub_drive.publish(stop)
        self.pub_opp_drive.publish(stop)
        
        while (self.scan is None or self.odom is None) and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.scan_buffer[0] = self.scan.copy()
        self.scan_buffer[1] = self.scan.copy()
        self.scan_buffer[2] = self.scan.copy()

        self.lap_count, self.second_lap_check = 0, False
        return self.scan_buffer.copy()

    def get_opp_pure_pursuit_action(self):
        if self.opp_odom is None: return 0.0, 1.5
        curr_x = self.opp_odom.pose.pose.position.x
        curr_y = self.opp_odom.pose.pose.position.y
        q = self.opp_odom.pose.pose.orientation
        curr_yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        num_wp = len(self.waypoints)
        for i in range(num_wp):
            idx = (self.current_wp_idx + i) % num_wp
            if np.linalg.norm(self.waypoints[idx] - np.array([curr_x, curr_y])) > self.lookahead_distance:
                self.current_wp_idx = idx
                self.target = self.waypoints[idx]
                break
        
        dx, dy = self.target[0] - curr_x, self.target[1] - curr_y
        local_y = -dx * np.sin(curr_yaw) + dy * np.cos(curr_yaw)
        steer = (2.0 * local_y) / (self.lookahead_distance**2)
        
        speed = 1.6 if abs(steer) < 0.1 else 1.1
        return float(np.clip(steer, -0.4, 0.4)), speed

    def step(self, action):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle, drive_msg.drive.speed = float(action[0]), float(action[1])
        self.pub_drive.publish(drive_msg)

        opp_steer, opp_speed = self.get_opp_pure_pursuit_action()
        opp_msg = AckermannDriveStamped()
        opp_msg.drive.steering_angle, opp_msg.drive.speed = opp_steer, opp_speed
        self.pub_opp_drive.publish(opp_msg)
        
        self.publish_debug_markers()

        for _ in range(10): rclpy.spin_once(self.node, timeout_sec=0.005)

        if self.scan is None: return self.scan_buffer.copy(), 0.0, False, {}

        self.scan_buffer[2] = self.scan_buffer[1]
        self.scan_buffer[1] = self.scan_buffer[0]
        self.scan_buffer[0] = self.scan.copy()

        reward = -1.3 + (float(action[1]) * 1.0)
        done = False

        if np.min(self.scan) < 0.25:
            reward, done = -100.0, True
            
            # Rakip aracın konumunu al (Eğer veri geldiyse)
            if self.opp_odom is not None:
                opp_x = self.opp_odom.pose.pose.position.x
                opp_y = self.opp_odom.pose.pose.position.y
                print(f"--- ÇARPIŞMA --- Rakip Konumu: X: {opp_x:.2f}, Y: {opp_y:.2f}", flush=True)
            else:
                print("--- ÇARPIŞMA --- (Rakip odom verisi alınamadı)", flush=True)

        if max(self.scan_buffer[0]) > 29.5:
            done = True  

        pos = self.odom.pose.pose.position
        dist_to_line = np.min(np.linalg.norm(self.finish_points - np.array([pos.x, pos.y]), axis=1))
        if dist_to_line < self.line_threshold and self.lap_count == 0:
            reward, self.lap_count = 500.0, 1
            print("🏁 TUR 1", flush=True)
        if self.lap_count == 1 and dist_to_line > self.second_lap_treshold:
            self.second_lap_check = True
        if dist_to_line < self.line_threshold and self.second_lap_check:
            reward, done = 1000.0, True
            print("🏁🏁 2 TUR BİTTİ!", flush=True)

        return self.scan_buffer.copy(), reward, done, {}

    def publish_debug_markers(self):
        f = Marker()
        f.header.frame_id = "map"
        f.ns = "finish"
        f.id = 202
        f.type = Marker.LINE_STRIP
        f.scale.x = 0.15
        f.color.g, f.color.a = 1.0, 1.0
        p1, p2 = Point(), Point()
        p1.x, p1.y = self.finish_A[0], self.finish_A[1]
        p2.x, p2.y = self.finish_B[0], self.finish_B[1]
        f.points = [p1, p2]
        self.marker_pub.publish(f)

def main():
    env = F110ROSEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=4096, batch_size=128)
    try:
        model.learn(total_timesteps=2000000)
        model.save("f1tenth_ppo_overtake_model")
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()