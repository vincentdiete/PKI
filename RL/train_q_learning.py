from collections import defaultdict
import numpy as np

from first_gym_environment import TwoGoalGridEnv


env = TwoGoalGridEnv(grid_size=5, max_steps=50)

# Q-Tabelle:
# Schlüssel: Zustand
# Wert: Array mit 4 Q-Werten, einer pro Aktion
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Hyperparameter
num_episodes = 5000
alpha = 0.1          # Lernrate
gamma = 0.95         # Zukunftsgewichtung
epsilon = 1.0        # Explorationswahrscheinlichkeit am Anfang
epsilon_min = 0.05
epsilon_decay = 0.995


def obs_to_state(obs):
    """
    Wandelt die Observation in einen nutzbaren Tabellen-Schlüssel um.
    NumPy-Arrays können nicht direkt als Dictionary-Key verwendet werden.
    Tupel dagegen schon.
    """
    return tuple(obs.tolist())

def greedy_action(q_values):
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return np.random.choice(best_actions)

for episode in range(num_episodes):
    obs, info = env.reset()
    state = obs_to_state(obs)

    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy-Strategie:
        # Mit Wahrscheinlichkeit epsilon zufällig handeln.
        # Sonst beste bekannte Aktion wählen.
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = greedy_action([q_table[state]])

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs_to_state(next_obs)

        done = terminated or truncated

        old_q = q_table[state][action]

        # Aktueller Q-Wert
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(q_table[next_state])

        q_table[state][action] = old_q + alpha * (target - old_q)

        state = next_state
        total_reward += reward

    # Epsilon langsam reduzieren
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        print(
            f"Episode {episode + 1}, "
            f"Total reward: {total_reward:.2f}, "
            f"Epsilon: {epsilon:.3f}, "
            f"Q-table states: {len(q_table)}"
        )


print("\nTraining abgeschlossen.")
print(f"Gelernte Zustände in Q-Tabelle: {len(q_table)}")


# Danach: Agent einmal anzeigen lassen
print("\n=== Gelernter Agent in einer Test-Episode ===")

obs, info = env.reset(seed=123)
state = obs_to_state(obs)

done = False
total_reward = 0

env.render()

while not done:
    action = np.argmax(q_table[state])

    obs, reward, terminated, truncated, info = env.step(action)
    state = obs_to_state(obs)

    total_reward += reward
    done = terminated or truncated

    print("Action:", action)
    print("Reward:", reward)
    print("Total reward:", total_reward)
    print("Info:", info)

    env.render()

print("Test-Episode beendet.")
print("Total reward:", total_reward)
print("Steps:", info["steps"])
print("Goals reached:", info["goals_reached"])