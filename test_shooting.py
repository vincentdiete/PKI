# test_shooting_agent.py

import numpy as np
from stable_baselines3 import SAC
from src.shooting_env import ShootingEnv

model = SAC.load("models/shooting_SAC_200k.zip")
env = ShootingEnv()

episodes = 1000
total_reward = 0.0
hits = 0
angle_errors = []

for _ in range(episodes):
    obs, info = env.reset()

    action, _ = model.predict(obs, deterministic=True)

    # Für Analyse: Richtung des Agenten mit echter Zielrichtung vergleichen
    shoot_vec = np.array(action, dtype=np.float32)
    shoot_dir = shoot_vec / (np.linalg.norm(shoot_vec) + 1e-8)

    target_vec = env.target_pos - env.player_pos
    target_dir = target_vec / (np.linalg.norm(target_vec) + 1e-8)

    dot = np.clip(np.dot(shoot_dir, target_dir), -1.0, 1.0)
    angle_error = np.degrees(np.arccos(dot))
    angle_errors.append(angle_error)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if reward > 1.0:
        hits += 1

print("Durchschnittlicher Reward:", total_reward / episodes)
print("Hit Rate:", hits / episodes)
print("Durchschnittlicher Winkelfehler:", np.mean(angle_errors), "Grad")
print("Median Winkelfehler:", np.median(angle_errors), "Grad")
