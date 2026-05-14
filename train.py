import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from src.environment import Environment
import pygame
import time


# Training
env = Environment()
model = stable_baselines3.SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=5000)
model.save("shooter_ppo_test")