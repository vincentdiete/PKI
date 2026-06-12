import numpy as np
from stable_baselines3 import SAC
from src.combat_env import CombatEnv
import os

MOVEMENT_MODEL_PATH = os.path.join("models", "movement_autoaim_SAC_500k.zip")
SHOOTING_MODEL_PATH = os.path.join("models", "shooting_SAC_200k.zip")

EPISODES = 20
DETERMINISTIC = True


def main():
    env = CombatEnv()

    movement_model = SAC.load(MOVEMENT_MODEL_PATH)
    shooting_model = SAC.load(SHOOTING_MODEL_PATH)

    episode_rewards = []
    episode_lengths = []
    episode_waves = []
    deaths = 0

    for episode in range(1, EPISODES + 1):
        obs, info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # 1) Movement-Agent bekommt die normale 17D-Observation
            move_action, _ = movement_model.predict(
                obs,
                deterministic=DETERMINISTIC,
            )

            # 2) Shooting-Agent bekommt seine eigene 3D-Observation
            shooting_obs, has_target = env.get_shooting_obs(only_visible=True)

            if has_target:
                shoot_action, _ = shooting_model.predict(
                    shooting_obs,
                    deterministic=DETERMINISTIC,
                )
            else:
                shoot_action = np.array([0.0, 0.0], dtype=np.float32)

            # 3) Beide Actions zusammensetzen: [move_x, move_y, shoot_x, shoot_y]
            combined_action = np.array(
                [
                    move_action[0],
                    move_action[1],
                    shoot_action[0],
                    shoot_action[1],
                ],
                dtype=np.float32,
            )

            obs, reward, terminated, truncated, info = env.step(combined_action)

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_waves.append(info.get("wave", 0))

        if terminated:
            deaths += 1

        print(
            f"Episode {episode:03d} | "
            f"reward={total_reward:.2f} | "
            f"steps={steps} | "
            f"waves={info.get('wave', 0)} | "
            f"terminated={terminated} | "
            f"truncated={truncated}"
        )

    print("\n--- Summary ---")
    print(f"Episodes: {EPISODES}")
    print(f"Avg reward: {np.mean(episode_rewards):.2f}")
    print(f"Avg episode length: {np.mean(episode_lengths):.2f}")
    print(f"Avg waves: {np.mean(episode_waves):.2f}")
    print(f"Death rate: {deaths / EPISODES:.2%}")

    env.close()


if __name__ == "__main__":
    main()
