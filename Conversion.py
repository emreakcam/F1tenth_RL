import torch
from stable_baselines3 import PPO

# 1. Modeli yükle
model = PPO.load("/sim_ws/robust_f1tenth_3million.zip", device="cpu")

# 2. Wrapper Sınıfı
# SB3'ün iç yapısı bazen karmaşık gelebilir, bu yüzden en temiz yolu kullanıyoruz
class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # Deterministic=True modunda aksiyon almak için .mode() kullanılır
        # Value (Değer) ve Action (Aksiyon) çıktılarını döndürüyoruz
        values = self.policy.predict_values(observation)
        actions = self.policy.get_distribution(observation).mode()
        return values, actions

# 3. Hazırlık
wrapped_model = PolicyWrapper(model.policy)
wrapped_model.eval()

# Girdi Boyutu: (Batch, Lidar_Noktası) -> (1, 1080)
# Eğer modelin tek bir lidar taramasıyla eğitildiyse stack=1 olur
dummy_input = torch.randn(1, 1080)

# 4. ONNX Export
torch.onnx.export(
    wrapped_model,
    dummy_input, 
    "3million_robust_single_lidar.onnx", 
    verbose=False,
    opset_version=11,
    input_names=['input'], 
    output_names=['value', 'action'], # Çıktıları net isimlendirmek hayat kurtarır
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'value': {0: 'batch_size'},
        'action': {0: 'batch_size'}
    }
)

print("Başarılı: 2million_single_lidar.onnx oluşturuldu!")