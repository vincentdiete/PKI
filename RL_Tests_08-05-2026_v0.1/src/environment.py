import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plot
from src.entities import Player, Monster

class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.width = 100
        self.height = 100
        self.max_steps = 100
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low = 100,
            high = 100,
            shape = (4,),
            dtype = np.float32
        )

        self.player = None
        self.monsters = []
        self.current_step = 0
        self.screen = None

    def _get_obs(self):
        return np.array([
            self.monsters[0].position[0] - self.player.position[0],
            self.monsters[0].position[1] - self.player.position[1],
            self.monsters[1].position[0] - self.player.position[0],
            self.monsters[1].position[1] - self.player.position[1]
        ], dtype = np.float32)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.heigth))

        self.screen.fill((0, 0, 0))

        pygame.draw.circle(self.screen, (255, 255, 255),
                        self.player.pos.astype(int), 8)

        for m in self.monsters:
            pygame.draw.circle(self.screen, (255, 0, 0),
                            m.pos.astype(int), 8)

        pygame.display.flip()

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.current_step = 0
        self.player = Player(50, 50)
        self.monsters = [
            Monster(np.random.randint(0, self.width), np.random.randint(0, self.height)),
            Monster(np.random.randint(0, self.width), np.random.randint(0, self.height))]
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        action_map = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
        }
        dx, dy = action_map[action]
        self.player.move(dx, dy, self.width, self.height)

        for m in self.monsters:
            m.move_toward(self.player)

        hit = any(
            np.linalg.norm(m.position - self.player.position) < 12
            for m in self.monsters
        )

        if hit:
            reward = -10.0
            terminated = True
        else:
            reward = 0.5
            terminated = False
        
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}


