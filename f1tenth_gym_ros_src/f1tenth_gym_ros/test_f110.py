import rclpy
from stable_baselines3 import PPO
import time
import os
# Kendi dosyanın adını 'f110_env' olarak varsayıyorum, eğer farklıysa import kısmını değiştir
# ya da Class tanımını bu dosyanın üstüne yapıştır.
from f1tenth_gym_ros.ros2_f110_env_takeover import F110ROSEnv

def main():
    # 1. Ortamı Başlat
    env = F110ROSEnv()
    
    # 2. Model Yolunu Belirle
    model_path = "/sim_ws/f1tenth_ppo_overtake_model.zip"
    
    if os.path.exists(model_path):
        print(f"--- Model Yükleniyor: {model_path} ---")
        model = PPO.load(model_path, env=env)
    else:
        print(f"HATA: {model_path} bulunamadı! Lütfen önce modeli eğitin.")
        return

    # 3. Test Parametreleri
    num_episodes = 10  # Kaç farklı senaryoda test edilecek
    
    print(f"--- Test Başlıyor: {num_episodes} Bölüm ---")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_counter = 0
        
        print(f"\n>> Bölüm {ep + 1} Başladı")
        
        while not done and rclpy.ok():
            # Modelden en iyi aksiyonu seç (deterministic=True)
            action, _states = model.predict(obs, deterministic=True)
            
            # Adım at
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            step_counter += 1
            
            # Test sırasında çok hızlı akmaması ve görsel takip için opsiyonel:
            # time.sleep(0.01) 

        print(f"Bölüm Bitti | Adım: {step_counter} | Toplam Ödül: {total_reward:.2f}")
        
    print("\n--- Tüm Testler Tamamlandı ---")
    env.node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()