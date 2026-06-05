import time
import stable_baselines3
from src.ros2 import TowerDefensePublisher
import os

publisher = TowerDefensePublisher()
obs, info = publisher.reset()
model_path = os.path.join(os.path.dirname(__file__), "models", "best", "best_model.zip")

model = stable_baselines3.SAC.load(model_path)


for step in range(10000000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = publisher.step(action)
    print(f"Step {step} | Action: {action} | Reward: {reward:.2f}")
    time.sleep(0.01)
    if terminated or truncated:
        print(f"RESET bei Step {step}!")
        obs, info = publisher.reset()

publisher.close()
