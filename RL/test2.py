import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridReachEnv(gym.Env):
    """
    Eine kleine 2D-Gymnasium-Umgebung.

    Aufgabe:
    Ein Agent bewegt sich auf einem 7x7-Gitter.
    Er startet zufällig und soll ein festes Ziel erreichen.
    Einige Felder sind Hindernisse.

    Aktionen:
        0 = hoch
        1 = rechts
        2 = runter
        3 = links

    Beobachtung:
        [agent_x, agent_y, target_x, target_y]

    Reward:
        - kleine Zeitstrafe pro Schritt
        + Belohnung, wenn der Agent näher ans Ziel kommt
        - Strafe bei Wand/Hindernis
        + große Belohnung bei Zielerreichung
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=7, max_steps=50):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Feste Hindernisse im Grid

        # Festes Ziel
        self.target_pos = {
            (6, 6),
            (6, 0),
            (0, 6),
            (0, 0),
        }
        
        # Vier diskrete Aktionen: hoch, rechts, runter, links
        self.action_space = spaces.Discrete(4)

        # Beobachtung:
        # agent_x, agent_y, target_x, target_y
        #
        # Jeder Wert liegt zwischen 0 und grid_size - 1.
        self.observation_space = spaces.MultiDiscrete(
            [grid_size, grid_size, grid_size, grid_size]
        )

        self.agent_pos = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """
        Startet eine neue Episode.
        Der Agent wird zufällig auf ein freies Feld gesetzt.
        """

        super().reset(seed=seed)

        self.current_step = 0

        free_cells = self._free_cells_without_target()

        start_index = self.np_random.integers(len(free_cells))
        self.agent_pos = free_cells[start_index]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Führt genau einen Schritt in der Umgebung aus.

        Input:
            action: Aktion des Agenten

        Output:
            observation: neue Beobachtung
            reward: Bewertung der Aktion
            terminated: Ziel erreicht?
            truncated: Zeitlimit erreicht?
            info: Zusatzinformationen
        """

        self.current_step += 1

        old_distance = self._manhattan_distance(self.agent_pos, self.target_pos)

        proposed_pos = self._move(self.agent_pos, action)

        hit_wall = not self._inside_grid(proposed_pos)
        hit_obstacle = proposed_pos in self.obstacles

        invalid_move = hit_wall or hit_obstacle

        if invalid_move:
            # Agent bleibt stehen, wenn er gegen Wand/Hindernis läuft
            new_pos = self.agent_pos
        else:
            new_pos = proposed_pos

        self.agent_pos = new_pos

        new_distance = self._manhattan_distance(self.agent_pos, self.target_pos)

        # ----------------------------
        # Reward Design
        # ----------------------------

        reward = 0.0

        # Kleine Strafe für jeden Schritt:
        # Der Agent soll nicht ewig herumirren.
        reward -= 0.1

        # Fortschrittsreward:
        # Wenn die Distanz kleiner wird, ist old_distance - new_distance positiv.
        # Wenn die Distanz größer wird, ist es negativ.
        reward += 0.5 * (old_distance - new_distance)

        # Strafe für ungültige Bewegung
        if invalid_move:
            reward -= 1.0

        # Ziel erreicht
        terminated = self.agent_pos == self.target_pos

        if terminated:
            reward += 10.0

        # Zeitlimit erreicht
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Gibt das aktuelle Grid in der Konsole aus.
        """

        for y in range(self.grid_size):
            row = []

            for x in range(self.grid_size):
                cell = (x, y)

                if cell == self.agent_pos:
                    row.append("A")
                elif cell == self.target_pos:
                    row.append("T")
                elif cell in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")

            print(" ".join(row))

        distance = self._manhattan_distance(self.agent_pos, self.target_pos)
        print(f"Step: {self.current_step}, Distance: {distance}")
        print()

    def close(self):
        pass

    def _get_observation(self):
        """
        Baut die Beobachtung für den Agenten.
        """

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        return np.array([ax, ay, tx, ty], dtype=np.int64)

    def _get_info(self):
        """
        Zusatzinformationen für Debugging und Auswertung.
        """

        return {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "distance": self._manhattan_distance(self.agent_pos, self.target_pos),
            "step": self.current_step,
            "is_success": self.agent_pos == self.target_pos,
        }

    def _move(self, pos, action):
        """
        Berechnet die neue Position, ohne sie direkt zu übernehmen.
        """

        x, y = pos

        if action == 0:      # hoch
            return (x, y - 1)
        elif action == 1:    # rechts
            return (x + 1, y)
        elif action == 2:    # runter
            return (x, y + 1)
        elif action == 3:    # links
            return (x - 1, y)
        else:
            raise ValueError(f"Ungültige Aktion: {action}")

    def _inside_grid(self, pos):
        """
        Prüft, ob eine Position innerhalb des Spielfelds liegt.
        """

        x, y = pos

        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _free_cells_without_target(self):
        """
        Liefert alle Zellen, auf denen der Agent starten darf.
        """

        free_cells = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = (x, y)

                if cell not in self.obstacles and cell != self.target_pos:
                    free_cells.append(cell)

        return free_cells

    @staticmethod
    def _manhattan_distance(a, b):
        """
        Manhattan-Distanz im Grid.
        """

        return abs(a[0] - b[0]) + abs(a[1] - b[1])


def choose_action(Q, observation, env, epsilon):
    """
    Epsilon-greedy Aktionswahl.

    Mit Wahrscheinlichkeit epsilon:
        zufällige Aktion = Exploration

    Mit Wahrscheinlichkeit 1 - epsilon:
        beste bekannte Aktion = Exploitation
    """

    if np.random.random() < epsilon:
        return env.action_space.sample()

    state = tuple(observation)

    return int(np.argmax(Q[state]))


def train_q_learning(
    env,
    episodes=8000,
    alpha=0.15,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
):
    """
    Trainiert einen tabellarischen Q-Learning-Agenten.
    """

    q_shape = tuple(env.observation_space.nvec) + (env.action_space.n,)
    Q = np.zeros(q_shape, dtype=np.float32)

    epsilon = epsilon_start

    episode_rewards = []
    episode_successes = []

    for episode in range(1, episodes + 1):
        observation, info = env.reset()

        done = False
        total_reward = 0.0
        success = False

        while not done:
            action = choose_action(Q, observation, env, epsilon)

            next_observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            success = terminated

            state = tuple(observation)
            next_state = tuple(next_observation)

            old_q_value = Q[state + (action,)]

            if terminated:
                td_target = reward
            else:
                best_future_q_value = np.max(Q[next_state])
                td_target = reward + gamma * best_future_q_value

            td_error = td_target - old_q_value

            # ----------------------------
            # Hier findet das Lernen statt.
            # ----------------------------
            Q[state + (action,)] = old_q_value + alpha * td_error

            observation = next_observation
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_successes.append(1 if success else 0)

        if episode % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            success_rate = np.mean(episode_successes[-500:])

            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Success Rate: {success_rate:5.2f} | "
                f"Epsilon: {epsilon:5.3f}"
            )

    return Q


def evaluate_agent(Q, env, episodes=20, render_first_episode=True):
    """
    Testet den gelernten Agenten ohne Exploration.
    """

    successes = 0

    for episode in range(episodes):
        observation, info = env.reset()
        done = False

        if episode == 0 and render_first_episode:
            print("Evaluation Episode 1:")
            env.render()

        while not done:
            state = tuple(observation)

            # Keine zufälligen Aktionen mehr:
            # Der Agent nimmt immer die beste gelernte Aktion.
            action = int(np.argmax(Q[state]))

            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            if episode == 0 and render_first_episode:
                print(f"Action: {action}")
                env.render()

        if terminated:
            successes += 1

    success_rate = successes / episodes

    print(f"Evaluation Success Rate: {success_rate:.2f}")


if __name__ == "__main__":
    env = GridReachEnv()

    print("Untrainierter Agent, eine zufällige Episode:")
    observation, info = env.reset()
    env.render()

    for _ in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Random Action: {action}, Reward: {reward:.2f}")
        env.render()

        if terminated or truncated:
            break

    print("Training startet...")
    Q = train_q_learning(env)

    print()
    print("Training abgeschlossen.")
    print()

    evaluate_agent(Q, env)