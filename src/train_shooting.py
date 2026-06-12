import os
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from src.shooting_env import ShootingEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = Monitor(ShootingEnv())

model = stable_baselines3.SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/shooting"
)

model.learn(total_timesteps=200_000)

model.save("models/shooting_SAC_200k")

print("Shooting-Modell gespeichert unter: models/shooting_SAC_200k.zip")