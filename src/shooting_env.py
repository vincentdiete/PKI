import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ShootingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.width = 6.0
        self.height = 6.0
        self.hit_radius = 0.35

        self.player_pos = np.array([3.0, 3.0], dtype=np.float32)
        self.target_pos = None

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            x = np.random.uniform(0.5, self.width - 0.5)
            y = np.random.uniform(0.5, self.height - 0.5)

            self.target_pos = np.array([x, y], dtype=np.float32)

            if np.linalg.norm(self.target_pos - self.player_pos) > 0.7:
                break

        return self._get_obs(), {}

    def _get_obs(self):
        rel = self.target_pos - self.player_pos
        dx = rel[0] / self.width
        dy = rel[1] / self.height
        dist = np.linalg.norm(rel) / np.sqrt(self.width ** 2 + self.height ** 2)

        return np.array([dx, dy, dist], dtype=np.float32)

    def step(self, action):
        shoot_vec = np.array(action, dtype=np.float32)
        norm = np.linalg.norm(shoot_vec)

        if norm < 1e-8:
            reward = -1.0
            terminated = True
            truncated = False
            return self._get_obs(), reward, terminated, truncated, {}

        shoot_dir = shoot_vec / norm

        target_vec = self.target_pos - self.player_pos
        t = np.dot(target_vec, shoot_dir)

        if t < 0:
            reward = -1.0
        else:
            closest_point = self.player_pos + t * shoot_dir
            dist_to_ray = np.linalg.norm(self.target_pos - closest_point)

            aim_quality = max(0.0, 1.0 - dist_to_ray / self.hit_radius)

            reward = aim_quality

            if dist_to_ray < self.hit_radius:
                reward += 1.0

        terminated = True
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}