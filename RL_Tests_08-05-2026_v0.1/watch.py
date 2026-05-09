from stable_baselines3 import PPO
from src.environment import Environment

env = Environment()
model = PPO.load("shooter_ppo")

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()