from collections import defaultdict
import pickle
import numpy as np

from first_env_ql.first_gym_environment import TwoGoalGridEnv


def obs_to_state(obs):
    return tuple(obs.tolist())


def greedy_action(q_values):
    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    return int(np.random.choice(best_actions))


def evaluate_agent(num_episodes=100, seed=None):
    env = TwoGoalGridEnv(grid_size=5, max_steps=100)

    with open("q_table.pkl", "rb") as file:
        loaded_q_table = pickle.load(file)

    q_table = defaultdict(
        lambda: np.zeros(env.action_space.n),
        loaded_q_table
    )

    successes = 0
    total_rewards = []
    total_steps = []
    unknown_state_counter = 0
    known_state_counter = 0

    rng = np.random.default_rng(seed)

    for episode in range(num_episodes):
        episode_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None

        obs, info = env.reset(seed=episode_seed)
        state = obs_to_state(obs)

        done = False
        episode_reward = 0

        while not done:
            if state in loaded_q_table:
                known_state_counter += 1
            else:
                unknown_state_counter += 1

            q_values = q_table[state]
            action = greedy_action(q_values)

            obs, reward, terminated, truncated, info = env.step(action)
            state = obs_to_state(obs)

            episode_reward += reward
            done = terminated or truncated

        if all(info["goals_reached"]):
            successes += 1

        total_rewards.append(episode_reward)
        total_steps.append(info["steps"])

    success_rate = successes / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print("\n=== Evaluation ===")
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Known states used: {known_state_counter}")
    print(f"Unknown states used: {unknown_state_counter}")

    if known_state_counter + unknown_state_counter > 0:
        unknown_ratio = unknown_state_counter / (known_state_counter + unknown_state_counter)
        print(f"Unknown state ratio: {unknown_ratio:.2%}")


if __name__ == "__main__":
    evaluate_agent(num_episodes=1000, seed=42)