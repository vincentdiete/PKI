from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from src.environment import Environment

# Einmalige Pruefung der Gymnasium-Kompatibilitaet
check_env(Environment(), warn=True)

train_env = Monitor(Environment())
eval_env = Monitor(Environment())

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best",
    log_path="./logs/eval",
    eval_freq=10_000,
    deterministic=True,
    render=False,
)
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./models/checkpoints",
    name_prefix="sac_enemy_planning",
)

model = SAC(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./logs",
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
)

model.learn(
    total_timesteps=100,
    callback=[eval_callback, checkpoint_callback],
    log_interval=10,
)

model.save("shooter_SAC")  # speichert als shooter_SAC.zip im aktuellen Verzeichnis
