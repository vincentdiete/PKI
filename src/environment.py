import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from src.entities import Player, Monster, Bullet, Goblin, Golem, Obstacle
import pygame
import time
from src.grid_map import GridMap

# Umgebung
class Environment(gymnasium.Env):
    def __init__(self):
        super(Environment, self).__init__()

        self.width = 6
        self.height = 6

        self.obstacles = [
            Obstacle(1.5, 1.5, 1.0, 1.0),
            Obstacle(4.5, 1.5, 1.0, 1.0),
            Obstacle(1.0, 4.0, 0.5, 0.5),
            Obstacle(4.0, 4.0, 1.0, 1.0)
        ]

        self.grid_map = GridMap(
            width=self.width,
            height=self.height,
            cell_size=0.25,
            obstacles=self.obstacles,
            obstacle_margin=0.15
        )
        self.episode_waves = []
        self.curriculum_threshold_waves = 2.0
        self.curriculum_level = 1
        self.episode_lengths = []
        self.curriculum_threshold = 1500
        self.max_steps = 3000
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
            )

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

        monster_high = [np.pi, 15.0] * 4
        wall_high = [6.0] * 4
        cooldown_high = [10.0]
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

        sorted_obstacles = sorted(enumerate(self.obstacles),
            key = lambda io: (np.linalg.norm(
                np.array([io[1].x + io[1].width/2, io[1].y + io[1].height/2]) - self.player.position
            ), io[0]))
        sorted_obstacles = [o for _, o in sorted_obstacles]


        def obstacle_rel(i):
                if i < len(sorted_obstacles):
                    cx = sorted_obstacles[i].x + sorted_obstacles[i].width / 2
                    cy = sorted_obstacles[i].y + sorted_obstacles[i].height / 2
                    dx = cx - self.player.position[0]
                    dy = cy - self.player.position[1]
                    dist = sorted_obstacles[i].distance_obstacle_to_player(
                    self.player.position[0], self.player.position[1]
                    )
                    theta = np.arctan2(dy, dx)
                    return [theta, dist]
                else:
                    return[0.0,0.0]

        sorted_monsters = sorted(enumerate(self.monsters),
            key = lambda im: (np.linalg.norm(im[1].position - self.player.position), im[0]))
        sorted_monsters = [m for _, m in sorted_monsters]
        
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
       
        if len(self.episode_waves) >= 20:
            recent_waves = self.episode_waves[-20:]
            avg_waves = np.mean(recent_waves)

            if avg_waves >= self.curriculum_threshold_waves and self.curriculum_level < 4:
                self.curriculum_level += 1

            self.episode_waves = self.episode_waves[-20:]
            self.episode_lengths = self.episode_lengths[-20:]

        self.current_step = 0
        self.wave = 0
        self.shoot_cooldown = 0
        self.bullets = []
        self.last_position = None
       
        margin = 1.5
        while True:
            px = np.random.uniform(margin, self.width - margin)
            py = np.random.uniform(margin, self.height - margin)
            if not any(o.contains_p(px, py, radius = 0.2) for o in self.obstacles):
                break
        self.player = Player(px, py)

        self.monsters = [
            self.spawn_monster_on_edge()
            for _ in range(self.curriculum_level)
        ]
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

        move_x, move_y, shoot_x, shoot_y = action

        # -------------------------
        # Player-Bewegung
        # -------------------------
        move_vec = np.array([move_x, move_y], dtype=np.float32)
        move_norm = np.linalg.norm(move_vec)

        prev_player_pos = self.player.position.copy()

        # Nur bewegen, wenn der Bewegungsvektor deutlich genug ist.
        # Dadurch kann der Agent auch stehen bleiben.
        if move_norm > 0.1:
            move_dir = move_vec / move_norm
            self.player.move(move_dir[0], move_dir[1], self.width, self.height)

        # Kollision mit Hindernissen: Bewegung rückgängig machen
        for o in self.obstacles:
            if o.contains_p(self.player.position[0], self.player.position[1], radius=0.15):
                self.player.position = prev_player_pos.copy()
                break

        # -------------------------
        # Schießen
        # -------------------------
        shoot_vec = np.array([shoot_x, shoot_y], dtype=np.float32)
        shoot_norm = np.linalg.norm(shoot_vec)

        # Nur schießen, wenn Cooldown frei ist und der Schussvektor groß genug ist.
        if self.shoot_cooldown == 0 and shoot_norm > 0.1:
            shoot_dir = shoot_vec / shoot_norm

            self.bullets.append(Bullet(
                self.player.position[0],
                self.player.position[1],
                shoot_dir[0],
                shoot_dir[1]
            ))

            self.shoot_cooldown = 10

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        for b in self.bullets:
            b.update()

        self.bullets = [b for b in self.bullets
                        if not b.out_of_bounds(self.width, self.height)
                        and not any(o.contains_p(b.position[0], b.position[1])
                        for o in self.obstacles)]

        remaining_bullets = []

        for b in self.bullets:
            bullet_hit = False

            for m in self.monsters:
                if np.linalg.norm(b.position - m.position) < 0.35:
                    m.hp -= b.damage
                    reward += 1.0  # direkter Treffer-Reward

                    if m.hp <= 0:
                        reward += 4.0  # direkter Kill-Reward

                    bullet_hit = True
                    break

            if not bullet_hit:
                remaining_bullets.append(b)

        self.bullets = remaining_bullets

        self.monsters = [m for m in self.monsters if m.hp > 0]

        if len(self.monsters) == 0:
            self.wave += 1
            monster_count = self.curriculum_level + (self.wave -1)
            for _ in range(monster_count):
              self.monsters.append(self.spawn_monster_on_edge())
            reward += 8.0

        for m in self.monsters:
            m.update(
                player=self.player,
                obstacles=self.obstacles,
                width=self.width,
                height=self.height,
                grid_map=self.grid_map
            )

        hit = any(
            np.linalg.norm(m.position - self.player.position) < 0.27
            for m in self.monsters
        )
       
        margin = min(
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1]
            )

        
        if margin < 0.5:
            reward-= 0.2

        obstacle_margin = min(
            o.distance_obstacle_to_player(self.player.position[0], self.player.position[1])
            for o in self.obstacles
        )

        if obstacle_margin < 0.3:
            reward -= 0.2

        distances = [self.player.position[0], self.width - self.player.position[0],
                     self.player.position[1], self.height - self.player.position[1]]

        distances_sorted = sorted(distances)
        if distances_sorted[0] < 0.5 and distances_sorted[1] < 0.5:
            reward -= 0.5

        if hit:
            reward -= 10
            terminated = True
        else:
            reward += 0.02
            terminated = False

        truncated = self.current_step >= self.max_steps

        info = {
            "monster_states": [m.state.name for m in self.monsters],
            "monster_path_lengths": [len(m.path) for m in self.monsters],
            "monster_waypoint_indices": [m.current_waypoint_index for m in self.monsters],
            "monster_blocked_reasons": [
                getattr(m, "blocked_reason", None)
                for m in self.monsters
            ],
        }

        return self._get_obs(), reward, terminated, truncated, info
