import numpy as np
from src.environment import Environment

env = Environment()
obs, info = env.reset()

# Kein Modell, nur zufällige Aktionen
for step in range(5000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 100 == 0:
        print(
            f"Step {step:04d} | "
            f"States: {info['monster_states']} | "
            f"Path lengths: {info['monster_path_lengths']} | "
            f"Waypoints: {info['monster_waypoint_indices']} | "
            f"Blocked reasons: {info['monster_blocked_reasons']} | "
            f"Reward: {reward:.2f}"
        )

    if terminated or truncated:
        obs, info = env.reset()