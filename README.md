# RL, Planung & ROS 2

Ein 2D-Shooter-Environment in dem ein RL-Agent (SAC) Bewegung und Schießen lernt. Gegner navigieren mit A*-Pfadplanung und Sichtlinienerkennung. ROS 2 verbindet die Komponenten als modulare Demo-Schicht.

---

## Projektstruktur

```
PKI/
├── train.py               # SAC-Training
├── watch.py               # Inferenz mit trainiertem Modell
├── test_planning.py       # A*-Pfadplanung isoliert testen
├── test_env_planning.py   # Planung im Environment testen
├── shooter_ppo.zip        # Gespeichertes PPO-Modell
└── src/
    ├── entities.py        # Player, Monster (mit A*), Bullet, Obstacle
    ├── environment.py     # Gymnasium-Environment
    ├── grid_map.py        # Diskretes Planungsgrid (40x40)
    ├── pathfinding.py     # A*-Algorithmus
    └── ros2.py            # ROS 2 Publisher-Node
```

---

## Architektur

```
Environment          RL-Agent (SAC)        ROS 2 Demo
State + Reward  -->  Move + Shoot     -->  Topics / Nodes

Enemy (A* + FSM)     Obstacles             Optional
Sichtlinie + Pfad    Cover + Kollision     Hierarchisches RL
```

**Training läuft lokal** in Gymnasium/Stable-Baselines3.
**ROS 2** dient ausschließlich für Inferenz und Demonstration.

---

## Gegnerverhalten

Monster nutzen eine FSM mit A*-Pfadplanung:

| State | Verhalten |
|---|---|
| `DIRECT_CHASE` | Freie Sichtlinie → direkt auf Spieler zu |
| `PATHFINDING` | Sichtlinie blockiert → A*-Pfad wird berechnet |
| `FOLLOW_PATH` | Monster folgt berechnetem Wegpunkt-Pfad |
| `BLOCKED` | Kein Pfad gefunden oder Kollision |
| `REACHED` | Spieler erreicht |

Das Planungsgrid hat `cell_size=0.25` → 40×40 Zellen auf einem 10×10-Feld.

---

## Installation

```bash
pip install gymnasium stable-baselines3 numpy
```

ROS 2 Setup (einmalig pro Terminal):
```bash
source /opt/ros/humble/setup.bash
```

---

## Verwendung

### Training
```bash
python train.py
```
Modell wird als `shooter_SAC.zip` gespeichert. Logs für TensorBoard in `./logs`.

```bash
tensorboard --logdir ./logs
```

### Inferenz
```bash
python watch.py
```

### ROS 2 Topics

| Topic | Typ | Inhalt |
|---|---|---|
| `/tower_defense/game_state` | `Float32MultiArray` | Spielzustand als Array |
| `/tower_defense/debug` | `String` | Spielzustand als JSON |

---

## Environment

### Observation Space (17-dimensional)

| Index | Inhalt |
|---|---|
| 0–7 | Winkel + Distanz zu den 4 nächsten Monstern |
| 8–11 | Wandabstände (links, rechts, unten, oben) |
| 12 | Schuss-Cooldown |
| 13–16 | Winkel + Distanz zu den 2 nächsten Hindernissen |

### Action Space (3-dimensional)

| Index | Wertebereich | Bedeutung |
|---|---|---|
| 0 | [-π, π] | Bewegungsrichtung |
| 1 | [-π, π] | Schussrichtung |
| 2 | [-1, 1] | Schuss-Trigger (> 0 = schießen) |

### Reward-Struktur

| Ereignis | Reward |
|---|---|
| Welle abgeschlossen | +15 |
| Überleben (pro Step) | +0.2 |
| Spieler getroffen | −10 |
| Wandnähe (margin < 1) | bis −15 |
| Ecke (zwei Wände < 0.8) | −8 |
| Hinderniskontakt (dist < 1) | bis −15 |

### Curriculum Learning

Start mit 1 Monster. Nach 20 Episoden mit Ø > 1500 Steps → nächste Stufe (max. 4 Monster).

---

## Roadmap

| Phase | Ziel | Status |
|---|---|---|
| 1 | RL-Baseline stabilisieren | In Arbeit |
| 2 | Observation erweitern (eigene Position, Gegner-HP, Sichtlinie) | Ausstehend |
| 3 | Hindernisse & Deckung | ✓ Implementiert |
| 4 | Gegnerplanung FSM + A* | ✓ Implementiert |
| 5 | ROS 2 Demo | Teilweise |
| 6 | Hierarchisches RL | Optional |

---

## Tech Stack

- **Gymnasium** + **Stable-Baselines3** — Environment & SAC
- **NumPy** — Berechnungen
- **ROS 2 (rclpy)** — Modulare Kommunikation
- **A\* / FSM** — Gegnerplanung
