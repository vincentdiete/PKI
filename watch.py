import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from src.environment import Environment
import pygame
import time

env = Environment()
model = stable_baselines3.SAC.load("shooter_ppo")

obs, _ = env.reset()

for _ in range(1000000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.001)
    if terminated or truncated:
        obs, _ = env.reset()