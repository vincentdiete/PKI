import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from src.entities import Player, Monster, Obstacle
from src.grid_map import GridMap


class Environment(gymnasium.Env):
    def __init__(self):
        super(Environment, self).__init__()

        self.width = 6
        self.height = 6

        self.obstacles = [
            Obstacle(1.5, 1.5, 1.0, 1.0),
            Obstacle(4.5, 1.5, 1.0, 1.0),
            Obstacle(1.0, 4.0, 0.5, 0.5),
            Obstacle(4.0, 4.0, 1.0, 1.0)
        ]

        self.grid_map = GridMap(
            width=self.width,
            height=self.height,
            cell_size=0.25,
            obstacles=self.obstacles,
            obstacle_margin=0.15
        )

        # Movement-only Curriculum: Schwierigkeit steigt über mehr Gegner beim Reset.
        self.curriculum_level = 1
        self.max_curriculum_level = 4
        self.episode_lengths = []
        self.max_steps = 3000

        # Movement-only Reward-Skala
        self.alive_reward = 0.005
        self.visible_target_reward = 0.02
        self.no_visible_target_penalty = -0.005

        self.too_close_penalty = -0.08
        self.good_distance_reward = 0.015
        self.too_far_penalty = -0.005

        self.wall_near_penalty = -0.04
        self.obstacle_near_penalty = -0.03
        self.corner_penalty = -0.12
        self.obstacle_collision_penalty = -0.10
        self.death_penalty = -12.0

        # Phase 1: Agent steuert nur Bewegung.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Bleibt für ROS2/Visualisierung vorhanden, wird in Movement-only aber nicht benutzt.
        self.bullets = []
        self.last_shoot_dir = np.array([0.0, 0.0], dtype=np.float32)
        self.wave = 0
        self.last_position = None
        self.screen = None
        self._fig = None
        self._ax = None

        # Observation Space V2:
        # 4 Monster     × (dx, dy, dist) = 12
        # 4 Wände       × Abstand        = 4
        # 4 Hindernisse × (dx, dy, dist) = 12
        # ------------------------------------
        # Gesamt                         = 28
        #
        # Alle Werte sind grob auf [-1, 1] bzw. [0, 1] normalisiert.
        monster_low = [-1.0, -1.0, 0.0] * 4
        wall_low = [0.0] * 4
        obstacle_low = [-1.0, -1.0, 0.0] * 4

        monster_high = [1.0, 1.0, 1.0] * 4
        wall_high = [1.0] * 4
        obstacle_high = [1.0, 1.0, 1.0] * 4

        self.observation_space = spaces.Box(
            low=np.array(monster_low + wall_low + obstacle_low, dtype=np.float32),
            high=np.array(monster_high + wall_high + obstacle_high, dtype=np.float32),
            shape=(28,),
            dtype=np.float32
        )

        self.player = None
        self.monsters = []
        self.current_step = 0

    def _get_obs(self):
        diagonal = np.sqrt(self.width ** 2 + self.height ** 2)

        # -------------------------
        # Monster: nächste 4 Gegner
        # -------------------------
        sorted_monsters = sorted(
            enumerate(self.monsters),
            key=lambda im: (
                np.linalg.norm(im[1].position - self.player.position),
                im[0]
            )
        )
        sorted_monsters = [m for _, m in sorted_monsters]

        def monster_rel(i):
            if i < len(sorted_monsters):
                rel = sorted_monsters[i].position - self.player.position
                dist = np.linalg.norm(rel)

                return [
                    rel[0] / self.width,
                    rel[1] / self.height,
                    dist / diagonal
                ]

            return [0.0, 0.0, 0.0]

        # -------------------------
        # Wände: normalisierte Abstände
        # -------------------------
        dist_left = float(self.player.position[0] / self.width)
        dist_right = float((self.width - self.player.position[0]) / self.width)
        dist_down = float(self.player.position[1] / self.height)
        dist_up = float((self.height - self.player.position[1]) / self.height)

        wall_obs = [
            dist_left,
            dist_right,
            dist_down,
            dist_up
        ]

        # -------------------------
        # Hindernisse: alle 4, sortiert nach Nähe
        # -------------------------
        sorted_obstacles = sorted(
            enumerate(self.obstacles),
            key=lambda io: (
                np.linalg.norm(
                    np.array(
                        [
                            io[1].x + io[1].width / 2,
                            io[1].y + io[1].height / 2
                        ],
                        dtype=np.float32
                    )
                    - self.player.position
                ),
                io[0]
            )
        )
        sorted_obstacles = [o for _, o in sorted_obstacles]

        def obstacle_rel(i):
            if i < len(sorted_obstacles):
                obstacle = sorted_obstacles[i]

                cx = obstacle.x + obstacle.width / 2
                cy = obstacle.y + obstacle.height / 2

                rel = np.array(
                    [
                        cx - self.player.position[0],
                        cy - self.player.position[1]
                    ],
                    dtype=np.float32
                )

                dist = obstacle.distance_obstacle_to_player(
                    self.player.position[0],
                    self.player.position[1]
                )

                return [
                    rel[0] / self.width,
                    rel[1] / self.height,
                    dist / diagonal
                ]

            return [0.0, 0.0, 0.0]

        obs = (
            monster_rel(0)
            + monster_rel(1)
            + monster_rel(2)
            + monster_rel(3)
            + wall_obs
            + obstacle_rel(0)
            + obstacle_rel(1)
            + obstacle_rel(2)
            + obstacle_rel(3)
        )

        return np.array(obs, dtype=np.float32)

    def get_frame(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        px, py = self.player.position.astype(int)
        for dx in range(-6, 6):
            for dy in range(-6, 6):
                if dx ** 2 + dy ** 2 < 36:
                    x, y = px + dx, py + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        frame[y, x] = [255, 255, 255]

        for m in self.monsters:
            mx, my = m.position.astype(int)
            for dx in range(-8, 8):
                for dy in range(-8, 8):
                    if dx ** 2 + dy ** 2 < 64:
                        x, y = mx + dx, my + dy
                        if 0 <= x < self.width and 0 <= y < self.height:
                            frame[y, x] = [255, 0, 0]

        return frame

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Movement-only Curriculum:
        # Kein Schießen, keine Kills, keine Waves.
        # Schwierigkeit steigt nur, wenn der Agent lange genug überlebt.
        if len(self.episode_lengths) >= 20:
            recent_lengths = self.episode_lengths[-20:]
            avg_length = np.mean(recent_lengths)

            if avg_length >= 0.8 * self.max_steps and self.curriculum_level < self.max_curriculum_level:
                self.curriculum_level += 1
                self.episode_lengths = []
            else:
                self.episode_lengths = self.episode_lengths[-20:]

        self.current_step = 0
        self.wave = 0
        self.shoot_cooldown = 0
        self.bullets = []
        self.last_position = None
        self.last_shoot_dir = np.array([0.0, 0.0], dtype=np.float32)

        margin = 1.5
        while True:
            px = np.random.uniform(margin, self.width - margin)
            py = np.random.uniform(margin, self.height - margin)
            if not any(o.contains_p(px, py, radius=0.2) for o in self.obstacles):
                break

        self.player = Player(px, py)

        self.monsters = [
            self.spawn_monster_on_edge()
            for _ in range(self.curriculum_level)
        ]

        return self._get_obs(), {}

    def spawn_monster_on_edge(self):
        side = np.random.randint(0, 4)
        if side == 0:
            x, y = np.random.uniform(0, self.width), 0
        elif side == 1:
            x, y = np.random.uniform(0, self.width), self.height
        elif side == 2:
            x, y = 0, np.random.uniform(0, self.height)
        else:
            x, y = self.width, np.random.uniform(0, self.height)

        return Monster(x, y)

    def has_clear_shot(self, target_position, step_size=0.05, radius=0.03):
        """
        Prüft, ob zwischen Player und Ziel ein Hindernis liegt.
        Wird in Movement-only als Positionssignal benutzt:
        freie Schusslinie = gute Position.
        """
        start = self.player.position
        end = target_position

        direction = end - start
        distance = np.linalg.norm(direction)

        if distance < 1e-8:
            return True

        direction = direction / distance
        steps = max(1, int(distance / step_size))

        for i in range(1, steps + 1):
            point = start + direction * step_size * i

            for obstacle in self.obstacles:
                if obstacle.contains_p(point[0], point[1], radius=radius):
                    return False

        return True

    def get_nearest_visible_monster(self):
        """
        Sucht den nächsten Gegner, der vom Player aus direkt beschießbar wäre.
        In Movement-only wird nicht geschossen; diese Funktion liefert nur Reward-Signal.
        """
        if len(self.monsters) == 0:
            return None

        visible_monsters = [
            m for m in self.monsters
            if self.has_clear_shot(m.position)
        ]

        if len(visible_monsters) == 0:
            return None

        return min(
            visible_monsters,
            key=lambda m: np.linalg.norm(m.position - self.player.position)
        )

    def get_nearest_monster_distance(self):
        if len(self.monsters) == 0:
            return None

        return min(
            np.linalg.norm(m.position - self.player.position)
            for m in self.monsters
        )

    def step(self, action):
        self.current_step += 1
        reward = 0.0

        move_x, move_y = action

        # -------------------------
        # Player-Bewegung
        # -------------------------
        move_vec = np.array([move_x, move_y], dtype=np.float32)
        move_norm = np.linalg.norm(move_vec)

        prev_player_pos = self.player.position.copy()

        if move_norm > 0.1:
            move_dir = move_vec / move_norm
            self.player.move(move_dir[0], move_dir[1], self.width, self.height)

        collided_with_obstacle = False

        for o in self.obstacles:
            if o.contains_p(self.player.position[0], self.player.position[1], radius=0.15):
                self.player.position = prev_player_pos.copy()
                collided_with_obstacle = True
                break

        if collided_with_obstacle:
            reward += self.obstacle_collision_penalty

        # -------------------------
        # Movement-only Reward
        # -------------------------
        # Kein Auto-Aim, keine Bullets, keine Kills, keine Waves.
        self.bullets = []
        self.shoot_cooldown = 0
        self.last_shoot_dir = np.array([0.0, 0.0], dtype=np.float32)

        visible_target = self.get_nearest_visible_monster()

        if visible_target is not None:
            reward += self.visible_target_reward
        else:
            reward += self.no_visible_target_penalty

        nearest_dist = self.get_nearest_monster_distance()

        if nearest_dist is not None:
            if nearest_dist < 0.7:
                reward += self.too_close_penalty
            elif 1.0 <= nearest_dist <= 2.5:
                reward += self.good_distance_reward
            elif nearest_dist > 4.5:
                reward += self.too_far_penalty

        # Gegner bewegen sich weiterhin. Genau dagegen soll Movement gelernt werden.
        for m in self.monsters:
            m.update(
                player=self.player,
                obstacles=self.obstacles,
                width=self.width,
                height=self.height,
                grid_map=self.grid_map
            )

        hit = any(
            np.linalg.norm(m.position - self.player.position) < 0.27
            for m in self.monsters
        )

        margin = min(
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1]
        )

        if margin < 0.5:
            reward += self.wall_near_penalty

        obstacle_margin = min(
            o.distance_obstacle_to_player(self.player.position[0], self.player.position[1])
            for o in self.obstacles
        )

        if obstacle_margin < 0.3:
            reward += self.obstacle_near_penalty

        distances = [
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1]
        ]

        distances_sorted = sorted(distances)
        if distances_sorted[0] < 0.5 and distances_sorted[1] < 0.5:
            reward += self.corner_penalty

        if hit:
            reward += self.death_penalty
            terminated = True
        else:
            reward += self.alive_reward
            terminated = False

        truncated = self.current_step >= self.max_steps

        info = {
            "curriculum_level": self.curriculum_level,
            "monster_states": [m.state.name for m in self.monsters],
            "monster_path_lengths": [len(m.path) for m in self.monsters],
            "monster_waypoint_indices": [m.current_waypoint_index for m in self.monsters],
            "monster_blocked_reasons": [
                getattr(m, "blocked_reason", None)
                for m in self.monsters
            ],
        }

        done = terminated or truncated

        if done:
            self.episode_lengths.append(self.current_step)

        return self._get_obs(), reward, terminated, truncated, info
