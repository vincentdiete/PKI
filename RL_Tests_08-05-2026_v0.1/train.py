# import libraries
from stable_baselines3 import PPO
from src.environment import Environment
from src.entities import Player
from src.entities import Monster


# Training
env = Environment()
model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=100000)

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

model.save("shooter_ppo")
