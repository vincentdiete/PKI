import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnemySurvivalEnv(gym.Env):
    """
    Gridworld-Survival-Umgebung.

    Agent steht in der Mitte eines 7x7-Grids.
    Zwei Gegner spawnen zufällig.
    Der Agent kann laufen oder in vier Richtungen schießen.
    Getroffene Gegner respawnen sofort.
    Gegner bewegen sich nach jeder Agentenaktion einen Schritt Richtung Agent.
    Ziel: möglichst viele Gegner abschießen und bis max_steps überleben.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=7, max_steps=100):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Aktionen:
        # 0 = hoch laufen
        # 1 = rechts laufen
        # 2 = runter laufen
        # 3 = links laufen
        # 4 = hoch schießen
        # 5 = rechts schießen
        # 6 = runter schießen
        # 7 = links schießen
        self.action_space = spaces.Discrete(8)

        # Beobachtung:
        # agent_x, agent_y,
        # enemy1_x, enemy1_y,
        # enemy2_x, enemy2_y
        self.observation_space = spaces.MultiDiscrete([
            grid_size, grid_size,
            grid_size, grid_size,
            grid_size, grid_size,
        ])

        self.agent_pos = None
        self.enemy_positions = None
        self.alive = True
        self.steps = 0
        self.kills = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.kills = 0
        self.alive = True

        # Agent startet in der Mitte
        center = self.grid_size // 2
        self.agent_pos = np.array([center, center], dtype=np.int32)

        self.enemy_positions = []
        while len(self.enemy_positions) < 2:
            self.enemy_positions.append(self._sample_enemy_position())

        observation = self._get_obs()
        info = {
            "kills": self.kills,
            "alive": self.alive,
        }

        return observation, info

    def step(self, action):
        self.steps += 1

        reward = -0.01
        hit_wall = False
        shot_hit = False
        killed_enemy_index = None

        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos.copy()

        # -------------------------
        # 1. Agentenaktion ausführen
        # -------------------------

        if action == 0:      # hoch
            new_pos[1] -= 1
        elif action == 1:    # rechts
            new_pos[0] += 1
        elif action == 2:    # runter
            new_pos[1] += 1
        elif action == 3:    # links
            new_pos[0] -= 1

        # Bewegungsaktion
        if action in [0, 1, 2, 3]:
            hit_wall = (
                new_pos[0] < 0 or
                new_pos[0] >= self.grid_size or
                new_pos[1] < 0 or
                new_pos[1] >= self.grid_size
            )

            if hit_wall:
                self.agent_pos = old_pos
                reward -= 0.10
            else:
                self.agent_pos = new_pos

        # Schussaktion
        elif action in [4, 5, 6, 7]:
            killed_enemy_index = self._get_hit_enemy_index(action)

            if killed_enemy_index is not None:
                shot_hit = True
                reward += 1.0
                self.kills += 1

                # Getroffener Gegner respawnt sofort
                self.enemy_positions[killed_enemy_index] = self._sample_enemy_position(
                    ignore_index=killed_enemy_index
                )
            else:
                reward -= 0.10

        # -----------------------------------
        # 2. Prüfen: Agent läuft in Gegner rein
        # -----------------------------------

        if self._agent_collides_with_enemy():
            self.alive = False
            reward -= 2.0

        # Wenn Agent schon tot ist, bewegen sich Gegner nicht mehr
        if self.alive:
            # -------------------------
            # 3. Gegner bewegen
            # -------------------------
            self._move_enemies_towards_agent(skip_index=killed_enemy_index)

            # -----------------------------------
            # 4. Prüfen: Gegner erreicht Agent
            # -----------------------------------
            if self._agent_collides_with_enemy():
                self.alive = False
                reward -= 2.0

        terminated = not self.alive
        truncated = self.steps >= self.max_steps

        observation = self._get_obs()

        info = {
            "steps": self.steps,
            "kills": self.kills,
            "hit_wall": bool(hit_wall),
            "shot_hit": bool(shot_hit),
            "alive": self.alive,
        }

        return observation, reward, terminated, truncated, info

    def _sample_enemy_position(self, ignore_index=None):
        """
        Erzeugt eine gültige Gegnerposition:
        - nicht auf dem Agenten
        - nicht auf einem anderen Gegner
        """

        while True:
            pos = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32,
            )

            if np.array_equal(pos, self.agent_pos):
                continue

            duplicate = False

            if self.enemy_positions is not None:
                for i, enemy_pos in enumerate(self.enemy_positions):
                    if ignore_index is not None and i == ignore_index:
                        continue

                    if np.array_equal(pos, enemy_pos):
                        duplicate = True
                        break

            if duplicate:
                continue

            return pos

    def _get_hit_enemy_index(self, action):
        """
        Gibt den Index des getroffenen Gegners zurück.
        Wenn mehrere Gegner in Schussrichtung stehen, wird der nächste getroffen.
        Wenn keiner getroffen wird, gibt die Methode None zurück.
        """

        hit_candidates = []

        for i, enemy_pos in enumerate(self.enemy_positions):
            same_column = self.agent_pos[0] == enemy_pos[0]
            same_row = self.agent_pos[1] == enemy_pos[1]

            enemy_above = enemy_pos[1] < self.agent_pos[1]
            enemy_right = enemy_pos[0] > self.agent_pos[0]
            enemy_below = enemy_pos[1] > self.agent_pos[1]
            enemy_left = enemy_pos[0] < self.agent_pos[0]

            hit = False

            if action == 4 and same_column and enemy_above:
                hit = True
            elif action == 5 and same_row and enemy_right:
                hit = True
            elif action == 6 and same_column and enemy_below:
                hit = True
            elif action == 7 and same_row and enemy_left:
                hit = True

            if hit:
                distance = abs(enemy_pos[0] - self.agent_pos[0]) + abs(enemy_pos[1] - self.agent_pos[1])
                hit_candidates.append((distance, i))

        if not hit_candidates:
            return None

        # Nächsten Gegner treffen
        hit_candidates.sort(key=lambda x: x[0])
        return hit_candidates[0][1]

    def _move_enemies_towards_agent(self, skip_index=None):
        """
        Jeder Gegner bewegt sich einen Schritt Richtung Agent.
        Wenn x- und y-Distanz beide ungleich 0 sind, wird zufällig entschieden,
        ob der Gegner horizontal oder vertikal näher kommt.
        """

        for i, enemy_pos in enumerate(self.enemy_positions):

            if skip_index is not None and i == skip_index:
                continue

            dx = self.agent_pos[0] - enemy_pos[0]
            dy = self.agent_pos[1] - enemy_pos[1]

            possible_axes = []

            if dx != 0:
                possible_axes.append("x")
            if dy != 0:
                possible_axes.append("y")

            if not possible_axes:
                continue

            axis = self.np_random.choice(possible_axes)

            if axis == "x":
                enemy_pos[0] += int(np.sign(dx))
            else:
                enemy_pos[1] += int(np.sign(dy))

    def _agent_collides_with_enemy(self):
        for enemy_pos in self.enemy_positions:
            if np.array_equal(self.agent_pos, enemy_pos):
                return True

        return False

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.enemy_positions[0][0],
            self.enemy_positions[0][1],
            self.enemy_positions[1][0],
            self.enemy_positions[1][1],
        ], dtype=np.int32)

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i, enemy_pos in enumerate(self.enemy_positions):
            x, y = enemy_pos
            grid[y][x] = f"E{i + 1}"

        if self.alive is not True:
            ax, ay = self.agent_pos
            grid[ay][ax] = "X"
        else:
            ax, ay = self.agent_pos
            grid[ay][ax] = "A"

        print()
        for row in grid:
            print(" ".join(row))
        print(f"Steps: {self.steps}, Kills: {self.kills}, Alive: {self.alive}")
        print()


if __name__ == "__main__":
    env = EnemySurvivalEnv(grid_size=7, max_steps=20)

    obs, info = env.reset(seed=42)
    print("Start observation:", obs)
    env.render()

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {t + 1}")
        print("Action:", action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", info)

        env.render()

        if terminated or truncated:
            break