from __future__ import annotations

import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.environment import Environment


def make_env(log_dir: str | None = None) -> Monitor:
    env = Environment()
    return Monitor(env, filename=log_dir)


def build_model(env: Monitor, tensorboard_log: str) -> SAC:
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )


def train(total_timesteps: int, run_dir: str, check: bool) -> None:
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models", "best"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs", "monitor", "train"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs", "monitor", "eval"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs", "eval"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs", "tensorboard"), exist_ok=True)

    if check:
        check_env(Environment(), warn=True)

    train_env = make_env(os.path.join(run_dir, "logs", "monitor", "train"))
    eval_env = make_env(os.path.join(run_dir, "logs", "monitor", "eval"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "models", "best"),
        log_path=os.path.join(run_dir, "logs", "eval"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(run_dir, "models", "checkpoints"),
        name_prefix="shooter_sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model = build_model(train_env, tensorboard_log=os.path.join(run_dir, "logs", "tensorboard"))
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(run_dir, "models", "shooter_sac_final"))
    train_env.close()
    eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on the tower-defense environment.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--run-dir", type=str, default="runs/shooter_sac")
    parser.add_argument("--no-check-env", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_timesteps=args.timesteps,
        run_dir=args.run_dir,
        check=not args.no_check_env,
    )
