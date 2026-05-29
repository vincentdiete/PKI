import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from src.entities import Player, Monster, Bullet, Goblin, Golem, Obstacle
import pygame
import time

# Umgebung
class Environment(gymnasium.Env):
    def __init__(self):
        super(Environment, self).__init__()

        self.obstacles = [
            Obstacle(2.5, 2.5, 1.0, 1.0),
            Obstacle(7.5, 2.5, 1.0, 1.0),
            Obstacle(1.5, 6.0, 0.5, 0.5),
            Obstacle(7.0, 7.0, 1.0, 1.0)
        ]

        self.curriculum_level = 1
        self.episode_lengths = []
        self.curriculum_threshold = 1500
        self.width = 10
        self.height = 10
        self.max_steps = 2000
        self.action_space = spaces.Box(
            low = np.array([-np.pi, -np.pi, -1.0]),
            high = np.array([np.pi, np.pi, 1.0]),
            dtype = np.float32)
        self.bullets = []
        self.shoot_cooldown = 0
        self.wave = 0
        self.last_position = None
        self.screen = None

        # Observation Space aufbauen
        monster_low = [-np.pi, 0.0] * 4
        wall_low = [0.0] * 4
        cooldown_low = [0.0]
        obstacle_low = [-np.pi, 0.0] * 2

        monster_high = [np.pi, 1500.0] * 4
        wall_high = [10.0] * 4
        cooldown_high = [30.0]
        obstacle_high = [np.pi, 20.0] * 2

        self.observation_space = spaces.Box(low= np.array(monster_low + wall_low + cooldown_low + obstacle_low),
                                            high= np.array(monster_high + wall_high + cooldown_high + obstacle_high),
                                            shape=(17,),
                                            dtype=np.float32
                                            )

        self.player = None
        self.monsters = []
        self.current_step = 0

    def _get_obs(self):

        sorted_obstacles = sorted(self.obstacles,
        key = lambda o: np.linalg.norm(
            np.array([o.x + o.width/2, o.y + o.height/2]) - self.player.position
        ))

        def obstacle_rel(i):
            if i < len(sorted_obstacles):
                cx = sorted_obstacles[i].x + sorted_obstacles[i].width / 2
                cy = sorted_obstacles[i].y + sorted_obstacles[i].height / 2
                dx = cx - self.player.position[0]
                dy = cy - self.player.position[1]
                dist = np.linalg.norm([dx, dy])
                theta = np.arctan2(dy, dx)
                return [theta, dist]
            else:
                return[0.0,0.0]

        sorted_monsters = sorted(self.monsters,
                               key = lambda m: np.linalg.norm(m.position - self.player.position))
        def monster_rel(i):
            if i < len(self.monsters):
                dx = sorted_monsters[i].position[0] - self.player.position[0]
                dy = sorted_monsters[i].position[1] - self.player.position[1]
                dist = np.linalg.norm([dx,dy])
                theta = np.arctan2(dy, dx)
                return [theta, dist]
            else:
                return [0.0, 0.0]

        dist_left = float(self.player.position[0])
        dist_right = float(self.width - self.player.position[0])
        dist_down = float(self.player.position[1])
        dist_up = float(self.height - self.player.position[1])

        return np.array(monster_rel(0) + monster_rel(1) + monster_rel(2) + monster_rel(3)
            + [dist_left, dist_right, dist_down, dist_up]
            + [float(self.shoot_cooldown)]
            + obstacle_rel(0) + obstacle_rel(1),
            dtype=np.float32)

    def get_frame(self):
      frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

      # Player — weißer Kreis
      px, py = self.player.position.astype(int)
      for dx in range(-6, 6):
          for dy in range(-6, 6):
              if dx**2 + dy**2 < 36:
                  x, y = px+dx, py+dy
                  if 0 <= x < self.width and 0 <= y < self.height:
                      frame[y, x] = [255, 255, 255]

      # Monster — rote Kreise
      for m in self.monsters:
          mx, my = m.position.astype(int)
          for dx in range(-8, 8):
              for dy in range(-8, 8):
                  if dx**2 + dy**2 < 64:
                      x, y = mx+dx, my+dy
                      if 0 <= x < self.width and 0 <= y < self.height:
                          frame[y, x] = [255, 0, 0]

      # Bullets — gelbe Punkte
      for b in self.bullets:
          bx, by = b.position.astype(int)
          if 0 <= bx < self.width and 0 <= by < self.height:
              frame[by, bx] = [255, 255, 0]

      return frame

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
       
        if self.current_step > 0:
            self.episode_lengths.append(self.current_step)

        if len(self.episode_lengths) >= 20:
            avg = np.mean(self.episode_lengths[-20:])
            if avg > self.curriculum_threshold and self.curriculum_level < 4:
                self.curriculum_level += 1
            self.episode_lengths = self.episode_lengths[-20:]
       
        self.current_step = 0
        self.wave = 0
        self.shoot_cooldown = 0
        self.bullets = []
        self.last_position = None
       
        margin = 3.5
        while True:
            px = np.random.uniform(margin, self.width - margin)
            py = np.random.uniform(margin, self.height - margin)
            if not any(o.contains_p(px, py, radius = 0.2) for o in self.obstacles):
                break
        self.player = Player(px, py)

        self.monsters = [self.spawn_monster_on_edge()
                         for _ in range(self.curriculum_level)]

        return self._get_obs(), {}

    def spawn_monster_on_edge(self):
        side = np.random.randint(0, 4)
        if side == 0:
            x,y = np.random.uniform(0, self.width), 0
        elif side == 1:
            x,y = np.random.uniform(0, self.width), self.height
        elif side == 2:
            x,y = 0, np.random.uniform(0, self.height)
        else:
            x,y = self.width, np.random.uniform(0, self.height)

        return Monster(x,y)

        '''
        -> Für Implementierung verschiedener Monster:
        roll = np.random.random()
        if roll < 0.25:
            return Goblin(x,y)
        elif roll > 0.5:
            return Monster(x,y)
        else:
            return Golem(x, y)
        '''

    def step(self, action):

        self.current_step += 1
        reward = 0.0

        move_theta, shoot_theta, shoot_trigger = action

        dx = np.cos(move_theta)
        dy = np.sin(move_theta)

        prev_player_pos = self.player.position.copy()

        self.player.move(dx, dy, self.width, self.height)

        for o in self.obstacles:
            if o.contains_p(self.player.position[0], self.player.position[1], radius = 0.05):
                self.player.position = prev_player_pos.copy()

        if shoot_trigger > 0 and self.shoot_cooldown == 0:
            self.bullets.append(Bullet(
                self.player.position[0],
                self.player.position[1],
                np.cos(shoot_theta),
                np.sin(shoot_theta)
            ))
            self.shoot_cooldown = 30

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        for b in self.bullets:
            b.update()

        self.bullets = [b for b in self.bullets
                        if not b.out_of_bounds(self.width, self.height)
                        and not any(o.contains_p(b.position[0], b.position[1])
                        for o in self.obstacles)]

        for b in self.bullets[:]:
            for m in self.monsters:
                if np.linalg.norm(b.position - m.position) < 0.1:
                    m.hp = m.hp - b.damage
                    self.bullets.remove(b)
                    break

        self.monsters = [m for m in self.monsters if m.hp > 0]

        if len(self.monsters) == 0:
            self.wave += 1
            monster_count = self.curriculum_level + (self.wave -1)
            for _ in range(monster_count):
              self.monsters.append(self.spawn_monster_on_edge())
            reward += 15

        for m in self.monsters:
            prev_pos = m.position.copy()
            m.move_toward(self.player)
            if any (o.contains_p(m.position[0], m.position[1], radius = 0.05)
            for o in self.obstacles):
                m.position = prev_pos

        hit = any(
            np.linalg.norm(m.position - self.player.position) < 0.27
            for m in self.monsters
        )

        margin = min(self.player.position[0], self.width - self.player.position[0],
                     self.player.position[1], self.height - self.player.position[1])
       
        if margin < 2:
            reward-= (1.0 - (margin/2)) * 15.0

        distances = [self.player.position[0], self.width - self.player.position[0],
                     self.player.position[1], self.height - self.player.position[1]]

        distances_sorted = sorted(distances)
        if distances_sorted[0] < 0.8 and distances_sorted[1] < 0.8:
            reward -= 8.0

        if hit:
            reward -= 10
            terminated = True
        else:
            reward += 0.2
            terminated = False

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}
