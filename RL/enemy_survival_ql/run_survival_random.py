from enemy_survival_env import EnemySurvivalEnv


env = EnemySurvivalEnv(grid_size=7, max_steps=30)

obs, info = env.reset(seed=13)

print("Start observation:", obs)
print("Start info:", info)
env.render()

done = False
total_reward = 0

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
    print("Info:", info)

    env.render()

print("Episode beendet.")
print("Total reward:", total_reward)
print("Final info:", info)