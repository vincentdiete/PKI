from collections import defaultdict, deque
import pickle
import numpy as np

from enemy_survival_env import EnemySurvivalEnv


def obs_to_state(obs):
    return tuple(obs.tolist())


def greedy_action(q_values):
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return int(np.random.choice(best_actions))


env = EnemySurvivalEnv(grid_size=7, max_steps=100)

q_table = defaultdict(lambda: np.zeros(env.action_space.n))

num_episodes = 50000

alpha = 0.15
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

recent_rewards = deque(maxlen=1000)
recent_kills = deque(maxlen=1000)
recent_steps = deque(maxlen=1000)
recent_deaths = deque(maxlen=1000)


for episode in range(1, num_episodes + 1):
    obs, info = env.reset()
    state = obs_to_state(obs)

    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = greedy_action(q_table[state])

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs_to_state(next_obs)

        done = terminated or truncated

        old_q = q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + gamma * np.max(q_table[next_state])

        q_table[state][action] = old_q + alpha * (target - old_q)

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    recent_rewards.append(total_reward)
    recent_kills.append(info["kills"])
    recent_steps.append(info["steps"])
    recent_deaths.append(0 if info["alive"] else 1)

    if episode % 1000 == 0:
        avg_reward = np.mean(recent_rewards)
        avg_kills = np.mean(recent_kills)
        avg_steps = np.mean(recent_steps)
        death_rate = np.mean(recent_deaths)

        print(
            f"Episode {episode}, "
            f"Avg reward last 1000: {avg_reward:.2f}, "
            f"Avg kills: {avg_kills:.2f}, "
            f"Avg steps alive: {avg_steps:.2f}, "
            f"Death rate: {death_rate:.2%}, "
            f"Epsilon: {epsilon:.3f}, "
            f"Q-table states: {len(q_table)}"
        )


with open("q_table_survival.pkl", "wb") as file:
    pickle.dump(dict(q_table), file)

print("\nTraining abgeschlossen.")
print(f"Gelernte Zustände: {len(q_table)}")
print("Q-Tabelle gespeichert als q_table_survival.pkl")


'''print("\n=== Testepisode ===")

obs, info = env.reset(seed=123)
state = obs_to_state(obs)

done = False
total_reward = 0

env.render()

while not done:
    action = greedy_action(q_table[state])

    obs, reward, terminated, truncated, info = env.step(action)
    state = obs_to_state(obs)

    total_reward += reward
    done = terminated or truncated

    print("Action:", action)
    print("Reward:", reward)
    print("Total reward:", total_reward)
    print("Info:", info)

    env.render()

print("Testepisode beendet.")
print("Total reward:", total_reward)
print("Steps:", info["steps"])
print("Kills:", info["kills"])
print("Alive:", info["alive"])'''