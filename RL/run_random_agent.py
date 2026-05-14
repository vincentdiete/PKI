from first_gym_environment import TwoGoalGridEnv


env = TwoGoalGridEnv(grid_size=5, max_steps=50)

num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    done = False

    print(f"\n=== Episode {episode + 1} ===")
    env.render()

    while not done:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        print("Action:", action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Total reward:", total_reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        env.render()
    ep = episode

    print(f"Episode {ep} finished after {info['steps']} steps")
    print(f"Total reward: {total_reward}")
    print(f"Goals reached: {info['goals_reached']}")