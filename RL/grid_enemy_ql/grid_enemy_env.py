import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridEnemyEnv(gym.Env):
    """
    Kleine Gridworld-Umgebung.

    Agent steht auf einem 5x5 Grid.
    Zwei Ziele werden zufällig platziert.
    Der Agent soll beide Ziele erreichen und abschießen.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=7, max_steps=50):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Aktionen:
        # 0 = hoch
        # 1 = rechts
        # 2 = runter
        # 3 = links
        # 4 = schuss hoch
        # 5 = schuss rechts
        # 6 = schuss runter
        # 7 = schuss links
        self.action_space = spaces.Discrete(8)

        # Beobachtung:
        # agent_x, agent_y,
        # goal1_x, goal1_y,
        # goal2_x, goal2_y,
        # goal1_eliminated, goal2_eliminated
        self.observation_space = spaces.MultiDiscrete([
            grid_size, grid_size,
            grid_size, grid_size,
            grid_size, grid_size,
            2, 2
        ])

        self.agent_pos = None
        self.goal_positions = None
        self.goals_eliminated = None
        self.alive = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.alive = True

        # Agent startet oben links
        self.agent_pos = np.array([0, 0], dtype=np.int32)

        # Zwei verschiedene Ziele erzeugen
        self.goal_positions = []
        while len(self.goal_positions) < 2:
            pos = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32
            )

            # Ziel darf nicht auf Agentenposition liegen
            if np.array_equal(pos, self.agent_pos):
                continue

            # Ziel darf nicht doppelt sein
            if any(np.array_equal(pos, g) for g in self.goal_positions):
                continue

            self.goal_positions.append(pos)

        self.goals_eliminated = [False, False]

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        self.steps += 1

        reward = -0.01
        hit_wall = False
        shot_hit = False

        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos.copy()

        # Bewegungsaktionen
        if action == 0:      # hoch
            new_pos[1] -= 1
        elif action == 1:    # rechts
            new_pos[0] += 1
        elif action == 2:    # runter
            new_pos[1] += 1
        elif action == 3:    # links
            new_pos[0] -= 1

        # Nur bei Bewegungsaktionen Wand prüfen
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

        # Schussaktionen
        elif action in [4, 5, 6, 7]:
            for i, goal_pos in enumerate(self.goal_positions):
                if self.goals_eliminated[i]:
                    continue

                same_column = self.agent_pos[0] == goal_pos[0]
                same_row = self.agent_pos[1] == goal_pos[1]

                enemy_above = goal_pos[1] < self.agent_pos[1]
                enemy_right = goal_pos[0] > self.agent_pos[0]
                enemy_below = goal_pos[1] > self.agent_pos[1]
                enemy_left = goal_pos[0] < self.agent_pos[0]

                if action == 4 and same_column and enemy_above:
                    self.goals_eliminated[i] = True
                    shot_hit = True
                    reward += 1.0
                    break

                elif action == 5 and same_row and enemy_right:
                    self.goals_eliminated[i] = True
                    shot_hit = True
                    reward += 1.0
                    break

                elif action == 6 and same_column and enemy_below:
                    self.goals_eliminated[i] = True
                    shot_hit = True
                    reward += 1.0
                    break

                elif action == 7 and same_row and enemy_left:
                    self.goals_eliminated[i] = True
                    shot_hit = True
                    reward += 1.0
                    break

            if not shot_hit:
                reward -= 0.10

        # Prüfen, ob der Agent in einen nicht eliminierten Gegner läuft
        for i, goal_pos in enumerate(self.goal_positions):
            if not self.goals_eliminated[i] and np.array_equal(self.agent_pos, goal_pos):
                self.alive = False
                reward -= 1.0

        terminated = all(self.goals_eliminated) or not self.alive
        truncated = self.steps >= self.max_steps

        observation = self._get_obs()

        info = {
            "steps": self.steps,
            "goals_eliminated": self.goals_eliminated.copy(),
            "hit_wall": bool(hit_wall),
            "shot_hit": shot_hit,
            "alive": self.alive,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Ziele einzeichnen
        for i, goal_pos in enumerate(self.goal_positions):
            x, y = goal_pos
            if self.goals_eliminated[i]:
                grid[y][x] = "x"
            else:
                grid[y][x] = f"G{i + 1}"

        # Agent einzeichnen
        ax, ay = self.agent_pos
        grid[ay][ax] = "A"

        print()
        for row in grid:
            print(" ".join(row))
        print()

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.goal_positions[0][0],
            self.goal_positions[0][1],
            self.goal_positions[1][0],
            self.goal_positions[1][1],
            int(self.goals_eliminated[0]),
            int(self.goals_eliminated[1])
        ], dtype=np.int32)


if __name__ == "__main__":
    env = GridEnemyEnv()

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