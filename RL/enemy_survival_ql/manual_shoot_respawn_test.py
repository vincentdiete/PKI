import numpy as np

from enemy_survival_env import EnemySurvivalEnv


env = EnemySurvivalEnv(grid_size=7, max_steps=20)

obs, info = env.reset(seed=42)

# Kontrollierten Zustand setzen
env.agent_pos = np.array([3, 3], dtype=np.int32)
env.enemy_positions = [
    np.array([3, 1], dtype=np.int32),  # Gegner über Agent
    np.array([6, 3], dtype=np.int32),  # Gegner rechts vom Agent
]
env.kills = 0
env.alive = True
env.steps = 0

print("=== Startzustand ===")
print("Observation:", env._get_obs())
env.render()

# 4 = Schuss hoch
action = 4
obs, reward, terminated, truncated, info = env.step(action)

print("=== Nach Schuss hoch ===")
print("Action:", action)
print("Observation:", obs)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)
env.render()

# 5 = Schuss rechts
action = 5
obs, reward, terminated, truncated, info = env.step(action)

print("=== Nach Schuss rechts ===")
print("Action:", action)
print("Observation:", obs)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)
env.render()