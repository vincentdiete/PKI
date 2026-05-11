import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from src.entities import Player, Monster, Bullet
import pygame
import time

# Umgebung
class Environment(gymnasium.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.width = 1000
        self.height = 1000
        self.max_steps = 1000000
        self.action_space = spaces.Discrete(12)
        self.bullets = []
        self.shoot_cooldown = 0
        self.wave = 0
        self.last_position = None
        self.screen = None

        # matplotlib state (statt pygame screen)
        # self._fig = None
        # self._ax = None

        self.observation_space = spaces.Box(
            low=-600,
            high=600,
            shape=(12,),
            dtype=np.float32
        )

        self.player = None
        self.monsters = []
        self.current_step = 0

    def _get_obs(self):

      sorted_monsters = sorted(self.monsters,
                               key = lambda m: np.linalg.norm(m.position - self.player.position))
      def monster_rel(i):
            if i < len(self.monsters):
                dx = sorted_monsters[i].position[0] - self.player.position[0]
                dy = sorted_monsters[i].position[1] - self.player.position[1]
                dist = np.linalg.norm([dx,dy])

                return [dx/dist if dist > 0 else 0.0,
                        dy/dist if dist > 0 else 0.0,
                        dist]
            else:
                return [0.0, 0.0, 0.0]

      return np.array(
            monster_rel(0) + monster_rel(1) + monster_rel(2) + monster_rel(3),
            dtype=np.float32
        )

    def _draw(self, ax):
        """Zeichnet den aktuellen Zustand auf eine matplotlib-Achse."""
        ax.clear()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_facecolor("black")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Player (weiß)
        if self.player is not None:
            ax.plot(self.player.position[0], self.player.position[1],
                    "o", color="white", markersize=8)

        # Monster (rot)
        for m in self.monsters:
            ax.plot(m.position[0], m.position[1],
                    "o", color="red", markersize=8)

        # Bullets (gelb)
        for b in self.bullets:
            ax.plot(b.position[0], b.position[1],
                    "o", color="yellow", markersize=4)

    '''
    def render(self):
        """Inline-Rendering für Colab (ersetzt pygame).

        Hinweis: Für jeden Step ein Bild anzuzeigen ist in Colab langsam.
        Für eine flüssige Visualisierung lieber `get_frame()` benutzen und am
        Ende eine Animation aus den Frames bauen (siehe Inference-Zelle).
        """
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(4, 4))
        self._draw(self._ax)
        self._fig.canvas.draw()
    '''

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))

        self.screen.fill((0,0,0))
        pygame.draw.circle(self.screen, (255,255,255),
            self.player.position.astype(int), 4)
        
        for m in self.monsters:
            pygame.draw.circle(self.screen, (255,0,0),
                m.position.astype(int), 4)

        for b in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 0),
                b.position.astype(int), 2)

        pygame.display.flip()

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
        self.current_step = 0
        self.player = Player(self.width/2, self.height/2)
        self.monsters = [self.spawn_monster_on_edge() for _ in range(4)]
        self.bullets = []
        self.shoot_cooldown = 0
        self.wave = 0
        self.last_position = 0

        return self._get_obs(), {}

    def spawn_monster_on_edge(self):
        side = np.random.randint(0, 4)
        if side == 0:
            return Monster(np.random.randint(0, self.width), 0)
        elif side == 1:
            return Monster(np.random.randint(0, self.width), self.height)
        elif side == 2:
            return Monster(0, np.random.randint(0, self.height))
        else:
            return Monster(self.width, np.random.randint(0, self.height))

    def step(self, action):

        self.current_step += 1
        reward = 0.0

        action_map = {
            0: ("move", 0, -1),
            1: ("move", 0, 1),
            2: ("move", -1, 0),
            3: ("move", 1, 0),

            4: ("shoot", 0, -1),
            5: ("shoot", 0, 1),
            6: ("shoot", 1, 0),
            7: ("shoot", -1, 0),

            8: ("shoot", 1, -1),
            9: ("shoot", 1, 1),
            10: ("shoot", -1, 1),
            11: ("shoot", -1, -1),
        }
        mode, dx, dy = action_map[int(action)]

        if self.last_position is not None:
            moved = np.linalg.norm(self.player.position - self.last_position)
            if moved < 0.5:
                reward = reward - 0.1
        self.last_position = self.player.position.copy()

        if mode == "move":
            self.player.move(dx, dy, self.width, self.height)

        elif mode == "shoot":
            if self.shoot_cooldown == 0:
                self.bullets.append(Bullet(
                    self.player.position[0],
                    self.player.position[1],
                    dx, dy
                ))
                self.shoot_cooldown = 10

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        for b in self.bullets:
            b.update()

        self.bullets = [b for b in self.bullets
                        if not b.out_of_bounds(self.width, self.height)]

        for b in self.bullets[:]:
            for m in self.monsters:
                if np.linalg.norm(b.position - m.position) < 10:
                    m.hp = m.hp - b.damage
                    self.bullets.remove(b)
                    break

        self.monsters = [m for m in self.monsters if m.hp > 0]

        if len(self.monsters) == 0:
            self.wave += 1
            for _ in range(self.wave + 3):
              self.monsters.append(self.spawn_monster_on_edge())
            reward += 15

        for m in self.monsters:
            m.move_toward(self.player)

        hit = any(
            np.linalg.norm(m.position - self.player.position) < 12
            for m in self.monsters
        )

        margin = min(self.player.position[0], self.width - self.player.position[0],
                     self.player.position[1], self.height - self.player.position[1])
        
        if margin < 20:
            reward-= 7

        if hit:
            reward -= 10
            terminated = True
        else:
            reward += 0.2
            terminated = False

        if len(self.monsters) < 4:
            reward += 3.0

        if len(self.monsters) == 0:
            terminated = True

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}