import gymnasium as gym
from gymnasium import spaces
import numpy as np


class OneDReachEnv(gym.Env):
    """
    Eine einfache 1D-Reach-Umgebung.

    Aufgabe:
    Ein Agent steht auf einer Linie und soll ein Ziel erreichen.
    Er kann pro Schritt nach links oder rechts gehen.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Zwei mögliche Aktionen:
        # 0 = nach links
        # 1 = nach rechts
        self.action_space = spaces.Discrete(2)

        # Beobachtung:
        # [Position des Agenten, Position des Ziels]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )

        self.agent_pos = 0.0
        self.target_pos = 0.0

        self.step_size = 0.5
        self.max_steps = 500
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Startzustand setzen
        self.agent_pos = 0.0
        self.target_pos = self.np_random.uniform(-5.0, 5.0)

        self.current_step = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.current_step += 1

        # Aktion ausführen
        if action == 0:
            self.agent_pos -= self.step_size
        elif action == 1:
            self.agent_pos += self.step_size
        else:
            raise ValueError(f"Ungültige Aktion: {action}")

        # Abstand zum Ziel berechnen
        distance = abs(self.agent_pos - self.target_pos)

        # Reward: je kleiner der Abstand, desto besser
        reward = -distance

        # Episode natürlich beendet: Ziel erreicht
        terminated = distance < 0.25

        # Episode künstlich beendet: Zeitlimit erreicht
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        distance = abs(self.agent_pos - self.target_pos)

        print(
            f"Step: {self.current_step:02d} | "
            f"Agent: {self.agent_pos:.2f} | "
            f"Target: {self.target_pos:.2f} | "
            f"Distance: {distance:.2f}"
        )

    def close(self):
        pass

    def _get_observation(self):
        return np.array(
            [self.agent_pos, self.target_pos],
            dtype=np.float32
        )

    def _get_info(self):
        return {
            "distance": abs(self.agent_pos - self.target_pos),
            "step": self.current_step
        }


if __name__ == "__main__":
    env = OneDReachEnv()

    observation, info = env.reset()

    print("Start observation:", observation)
    print("Start info:", info)
    print()

    for _ in range(100):
        # Zufällige Aktion wählen
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward:.2f}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        env.render()
        print("-" * 50)

        if terminated or truncated:
            print("Episode beendet. Neue Episode startet.")
            observation, info = env.reset()
            print("Neue observation:", observation)
            print()