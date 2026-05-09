from grid_enemy_env import GridEnemyEnv
import numpy as np


env = GridEnemyEnv(grid_size=5, max_steps=20)

obs, info = env.reset(seed=42)

# Für kontrollierten Test überschreiben wir die zufälligen Gegnerpositionen
env.agent_pos = np.array([2, 2], dtype=np.int32)
env.goal_positions = [
    np.array([2, 0], dtype=np.int32),  # Gegner über Agent
    np.array([4, 2], dtype=np.int32),  # Gegner rechts vom Agent
]
env.goals_eliminated = [False, False]
env.alive = True

print("Start observation:", env._get_obs())
env.render()

# Aktionen:
# 4 = Schuss hoch
# 5 = Schuss rechts
actions = [4, 5]

for action in actions:
    obs, reward, terminated, truncated, info = env.step(action)

    print("Action:", action)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    env.render()

    if terminated or truncated:
        break