import numpy as np

from enemy_survival_env import EnemySurvivalEnv


env = EnemySurvivalEnv(grid_size=7, max_steps=10)

obs, info = env.reset(seed=123)

# Kontrollierter Zustand
env.agent_pos = np.array([3, 3], dtype=np.int32)
env.enemy_positions = [
    np.array([0, 3], dtype=np.int32),  # links vom Agenten
    np.array([3, 6], dtype=np.int32),  # unter dem Agenten
]
env.kills = 0
env.alive = True
env.steps = 0

print("=== Startzustand ===")
env.render()

# Wir schießen absichtlich daneben, damit Agent stehen bleibt.
# Danach sollten sich die Gegner Richtung Agent bewegen.
for step in range(5):
    action = 4  # Schuss hoch, wahrscheinlich Fehlschuss

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"=== Step {step + 1} ===")
    print("Action:", action)
    print("Reward:", reward)
    print("Observation:", obs)
    print("Info:", info)
    env.render()

    if terminated or truncated:
        break