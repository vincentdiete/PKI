from collections import defaultdict
import pickle
import numpy as np

from grid_enemy_env import GridEnemyEnv


def obs_to_state(obs):
    return tuple(obs.tolist())


def greedy_action(q_values):
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return int(np.random.choice(best_actions))


def evaluate(num_episodes=1000, seed=42):
    env = GridEnemyEnv(grid_size=7, max_steps=50)

    with open("q_table_enemy.pkl", "rb") as file:
        loaded_q_table = pickle.load(file)

    q_table = defaultdict(
        lambda: np.zeros(env.action_space.n),
        loaded_q_table
    )

    rng = np.random.default_rng(seed)

    successes = 0
    deaths = 0
    truncations = 0
    total_rewards = []
    total_steps = []

    known_states = 0
    unknown_states = 0

    for _ in range(num_episodes):
        episode_seed = int(rng.integers(0, 1_000_000))
        obs, info = env.reset(seed=episode_seed)
        state = obs_to_state(obs)

        done = False
        episode_reward = 0

        while not done:
            if state in loaded_q_table:
                known_states += 1
            else:
                unknown_states += 1

            action = greedy_action(q_table[state])

            obs, reward, terminated, truncated, info = env.step(action)
            state = obs_to_state(obs)

            episode_reward += reward
            done = terminated or truncated

        if all(info["goals_eliminated"]):
            successes += 1

        if not info["alive"]:
            deaths += 1

        if truncated:
            truncations += 1

        total_rewards.append(episode_reward)
        total_steps.append(info["steps"])

    print("\n=== Enemy Agent Evaluation ===")
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {successes / num_episodes:.2%}")
    print(f"Death rate: {deaths / num_episodes:.2%}")
    print(f"Truncation rate: {truncations / num_episodes:.2%}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.2f}")

    total_state_uses = known_states + unknown_states
    print(f"Known states used: {known_states}")
    print(f"Unknown states used: {unknown_states}")
    print(f"Unknown state ratio: {unknown_states / total_state_uses:.2%}")


if __name__ == "__main__":
    evaluate(num_episodes=1000, seed=42)