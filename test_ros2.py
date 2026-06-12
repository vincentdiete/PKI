import os
import time
import numpy as np
from stable_baselines3 import SAC

from src.ros2 import TowerDefensePublisher


MOVEMENT_MODEL_PATH = os.path.join("models", "movement_autoaim_SAC_500k.zip")
SHOOTING_MODEL_PATH = os.path.join("models", "shooting_SAC_200k.zip")
SLEEP_SECONDS = 0.01
DETERMINISTIC = True


def main():
    publisher = TowerDefensePublisher()
    obs, info = publisher.reset()

    movement_model = SAC.load(MOVEMENT_MODEL_PATH)
    shooting_model = SAC.load(SHOOTING_MODEL_PATH)

    try:
        for step in range(10_000_000):
            move_action, _ = movement_model.predict(
                obs,
                deterministic=DETERMINISTIC,
            )

            shooting_obs, has_target = publisher.env.get_shooting_obs(only_visible=True)
            if has_target:
                shoot_action, _ = shooting_model.predict(
                    shooting_obs,
                    deterministic=DETERMINISTIC,
                )
            else:
                shoot_action = np.array([0.0, 0.0], dtype=np.float32)

            combined_action = np.array(
                [
                    move_action[0],
                    move_action[1],
                    shoot_action[0],
                    shoot_action[1],
                ],
                dtype=np.float32,
            )

            obs, reward, terminated, truncated, info = publisher.step(combined_action)

            print(
                f"Step {step} | "
                f"move={move_action} | "
                f"shoot={shoot_action} | "
                f"reward={reward:.2f} | "
                f"wave={info.get('wave', 0)} | "
                f"monsters={info.get('monster_count', 0)}"
            )

            time.sleep(SLEEP_SECONDS)

            if terminated or truncated:
                print(f"RESET bei Step {step}! terminated={terminated}, truncated={truncated}")
                obs, info = publisher.reset()

    finally:
        publisher.close()


if __name__ == "__main__":
    main()
