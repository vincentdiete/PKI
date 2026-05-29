# Verbesserte RL-Dateien

Diese Version ist ein kompletter Umbau gegenüber dem ursprünglichen Stand.

## Wichtiger Breaking Change
Der Action-Space ist nicht mehr `[move_theta, shoot_theta, shoot_trigger]`, sondern:

```python
[move_x, move_y, shoot_x, shoot_y]
```

Alte SAC-Modelle sind damit nicht kompatibel und müssen neu trainiert werden.

## Wichtigste Änderungen

- Survival-Reward von ehemals starkem Schritt-Reward auf `0.01` reduziert.
- Direkter Bullet-Hit-Reward: `+1.0`.
- Direkter Kill-Reward: `+4.0`.
- Wave-Clear-Reward reduziert auf `+8.0`.
- Extreme Wand-/Hindernis-Nähe-Strafen entfernt.
- Nur echte Hinderniskollision gibt eine kleine Strafe.
- Episodenlänge auf `max_steps = 3000` begrenzt.
- Observation vollständig normalisiert und ohne Winkel-Sprung bei `-pi/pi` aufgebaut.
- Monster-Observation nutzt `dx`, `dy`, `dist`, `hp`, `presence`.
- Bullet-Hit-Erkennung nutzt Segment-Kollision statt nur Punktposition nach dem Update.
- Trefferadius auf `0.20` erhöht.
- Curriculum steigt nach Kills/Wave-Clears, nicht nach bloßer Episodenlänge.
- Monster-Replanning throttled: nicht mehr bei jeder kleinen Zieländerung.
- `train.py` nutzt `check_env`, `Monitor`, getrennte Eval-Env, `EvalCallback`, `CheckpointCallback`.
- Unnötige Imports aus den Core-Dateien entfernt.

## Empfohlener Start

```bash
python train.py --timesteps 1000000 --run-dir runs/shooter_sac_v2
```

Für einen schnellen technischen Check:

```bash
python train.py --timesteps 10000 --run-dir runs/debug_v2
```

Wenn `check_env` während schneller Experimente nervt:

```bash
python train.py --timesteps 10000 --run-dir runs/debug_v2 --no-check-env
```
