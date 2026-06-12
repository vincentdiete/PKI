import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from src.entities import Player, Monster, Bullet, Obstacle
from src.grid_map import GridMap


class CombatEnv(gymnasium.Env):
    """
    Kombinierte Combat-Umgebung für die Evaluation von zwei getrennten Agenten:

    - Movement-Agent steuert action[0:2] = move_x, move_y
    - Shooting-Agent steuert action[2:4] = shoot_x, shoot_y

    Wichtig:
    Diese Klasse ist bewusst getrennt von der Movement-only-Environment.
    Sie aktiviert wieder Bullets, Kills und Waves, ohne Auto-Aim zu benutzen.
    """

    def __init__(self):
        super().__init__()

        self.width = 6.0
        self.height = 6.0

        self.obstacles = [
            Obstacle(1.5, 1.5, 1.0, 1.0),
            Obstacle(4.5, 1.5, 1.0, 1.0),
            Obstacle(1.0, 4.0, 0.5, 0.5),
            Obstacle(4.0, 4.0, 1.0, 1.0),
        ]

        self.grid_map = GridMap(
            width=self.width,
            height=self.height,
            cell_size=0.25,
            obstacles=self.obstacles,
            obstacle_margin=0.15,
        )

        # Schwierigkeit / Ablauf
        self.curriculum_level = 1
        self.max_steps = 3000
        self.wave = 0
        self.current_step = 0

        # Shooting
        self.bullets = []
        self.shoot_cooldown = 0
        self.shoot_cooldown_steps = 20
        self.last_shoot_dir = np.array([0.0, 0.0], dtype=np.float32)
        self.hit_radius = 0.35

        # Reward-Skala für Combat-Auswertung
        self.alive_reward = 0.001
        self.visible_target_reward = 0.01
        self.no_visible_target_penalty = -0.002

        self.too_close_penalty = -0.05
        self.good_distance_reward = 0.005
        self.too_far_penalty = -0.002

        self.wall_near_penalty = -0.03
        self.obstacle_near_penalty = -0.03
        self.corner_penalty = -0.08
        self.obstacle_collision_penalty = -0.10

        self.kill_reward = 4.0
        self.wave_clear_reward = 5.0
        self.death_penalty = -12.0

        # Kombinierte Action: move_x, move_y, shoot_x, shoot_y
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Movement-Observation bleibt kompatibel zum Movement-only-Agenten.
        monster_low = [-np.pi, 0.0] * 4
        wall_low = [0.0] * 4
        cooldown_low = [0.0]
        obstacle_low = [-np.pi, 0.0] * 2

        monster_high = [np.pi, 15.0] * 4
        wall_high = [6.0] * 4
        cooldown_high = [20.0]
        obstacle_high = [np.pi, 20.0] * 2

        self.observation_space = spaces.Box(
            low=np.array(monster_low + wall_low + cooldown_low + obstacle_low, dtype=np.float32),
            high=np.array(monster_high + wall_high + cooldown_high + obstacle_high, dtype=np.float32),
            shape=(17,),
            dtype=np.float32,
        )

        self.player = None
        self.monsters = []
        self.last_position = None
        self.screen = None
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs(self):
        """
        17D Movement-Observation, kompatibel zum Movement-only-Agenten.
        """
        sorted_obstacles = sorted(
            enumerate(self.obstacles),
            key=lambda io: (
                np.linalg.norm(
                    np.array(
                        [
                            io[1].x + io[1].width / 2,
                            io[1].y + io[1].height / 2,
                        ],
                        dtype=np.float32,
                    )
                    - self.player.position
                ),
                io[0],
            ),
        )
        sorted_obstacles = [o for _, o in sorted_obstacles]

        def obstacle_rel(i):
            if i < len(sorted_obstacles):
                cx = sorted_obstacles[i].x + sorted_obstacles[i].width / 2
                cy = sorted_obstacles[i].y + sorted_obstacles[i].height / 2
                dx = cx - self.player.position[0]
                dy = cy - self.player.position[1]
                dist = sorted_obstacles[i].distance_obstacle_to_player(
                    self.player.position[0],
                    self.player.position[1],
                )
                theta = np.arctan2(dy, dx)
                return [theta, dist]
            return [0.0, 0.0]

        sorted_monsters = sorted(
            enumerate(self.monsters),
            key=lambda im: (
                np.linalg.norm(im[1].position - self.player.position),
                im[0],
            ),
        )
        sorted_monsters = [m for _, m in sorted_monsters]

        def monster_rel(i):
            if i < len(sorted_monsters):
                dx = sorted_monsters[i].position[0] - self.player.position[0]
                dy = sorted_monsters[i].position[1] - self.player.position[1]
                dist = np.linalg.norm([dx, dy])
                theta = np.arctan2(dy, dx)
                return [theta, dist]
            return [0.0, 0.0]

        dist_left = float(self.player.position[0])
        dist_right = float(self.width - self.player.position[0])
        dist_down = float(self.player.position[1])
        dist_up = float(self.height - self.player.position[1])

        return np.array(
            monster_rel(0)
            + monster_rel(1)
            + monster_rel(2)
            + monster_rel(3)
            + [dist_left, dist_right, dist_down, dist_up]
            + [float(self.shoot_cooldown)]
            + obstacle_rel(0)
            + obstacle_rel(1),
            dtype=np.float32,
        )

    def get_shooting_obs(self, only_visible=True):
        """
        3D Shooting-Observation, kompatibel zur ShootingEnv:
        [dx / width, dy / height, distance / diagonal]

        Rückgabe:
            shooting_obs, has_target

        has_target=False bedeutet: kein sinnvoll beschießbarer Gegner vorhanden.
        Dann sollte der kombinierende Code shoot_action = [0, 0] setzen.
        """
        if only_visible:
            target = self.get_nearest_visible_monster()
        else:
            target = self.get_nearest_monster()

        if target is None:
            return np.zeros(3, dtype=np.float32), False

        rel = target.position - self.player.position
        dist = np.linalg.norm(rel)
        diagonal = np.sqrt(self.width ** 2 + self.height ** 2)

        obs = np.array(
            [
                rel[0] / self.width,
                rel[1] / self.height,
                dist / diagonal,
            ],
            dtype=np.float32,
        )
        return obs, True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def spawn_monster_on_edge(self):
        side = np.random.randint(0, 4)

        if side == 0:
            x, y = np.random.uniform(0, self.width), 0.0
        elif side == 1:
            x, y = np.random.uniform(0, self.width), self.height
        elif side == 2:
            x, y = 0.0, np.random.uniform(0, self.height)
        else:
            x, y = self.width, np.random.uniform(0, self.height)

        return Monster(x, y)

    def has_clear_shot(self, target_position, step_size=0.05, radius=0.03):
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

    def get_nearest_monster(self):
        if len(self.monsters) == 0:
            return None

        return min(
            self.monsters,
            key=lambda m: np.linalg.norm(m.position - self.player.position),
        )

    def get_nearest_visible_monster(self):
        visible_monsters = [
            m for m in self.monsters
            if self.has_clear_shot(m.position)
        ]

        if len(visible_monsters) == 0:
            return None

        return min(
            visible_monsters,
            key=lambda m: np.linalg.norm(m.position - self.player.position),
        )

    def get_nearest_monster_distance(self):
        nearest = self.get_nearest_monster()
        if nearest is None:
            return None
        return float(np.linalg.norm(nearest.position - self.player.position))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

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

    def step(self, action):
        self.current_step += 1
        reward = 0.0

        move_x, move_y, shoot_x, shoot_y = np.asarray(action, dtype=np.float32)

        # -------------------------
        # Movement
        # -------------------------
        move_vec = np.array([move_x, move_y], dtype=np.float32)
        move_norm = np.linalg.norm(move_vec)

        prev_player_pos = self.player.position.copy()

        if move_norm > 0.1:
            move_dir = move_vec / move_norm
            self.player.move(move_dir[0], move_dir[1], self.width, self.height)

        collided_with_obstacle = False
        for obstacle in self.obstacles:
            if obstacle.contains_p(self.player.position[0], self.player.position[1], radius=0.15):
                self.player.position = prev_player_pos.copy()
                collided_with_obstacle = True
                break

        if collided_with_obstacle:
            reward += self.obstacle_collision_penalty

        # -------------------------
        # Shooting: keine Auto-Aim-Logik, nur Action des Shooting-Agenten
        # -------------------------
        shoot_vec = np.array([shoot_x, shoot_y], dtype=np.float32)
        shoot_norm = np.linalg.norm(shoot_vec)

        if self.shoot_cooldown == 0 and shoot_norm > 1e-8:
            shoot_dir = shoot_vec / shoot_norm
            self.last_shoot_dir = shoot_dir.copy()

            self.bullets.append(
                Bullet(
                    self.player.position[0],
                    self.player.position[1],
                    shoot_dir[0],
                    shoot_dir[1],
                )
            )
            self.shoot_cooldown = self.shoot_cooldown_steps

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # -------------------------
        # Positionierungsreward
        # -------------------------
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

        # -------------------------
        # Bullet Update und Hit=Kill
        # -------------------------
        for bullet in self.bullets:
            bullet.update()

        self.bullets = [
            bullet for bullet in self.bullets
            if not bullet.out_of_bounds(self.width, self.height)
            and not any(
                obstacle.contains_p(bullet.position[0], bullet.position[1])
                for obstacle in self.obstacles
            )
        ]

        remaining_bullets = []
        killed_monsters = []

        for bullet in self.bullets:
            bullet_hit = False

            for monster in self.monsters:
                if monster in killed_monsters:
                    continue

                if np.linalg.norm(bullet.position - monster.position) < self.hit_radius:
                    reward += self.kill_reward
                    killed_monsters.append(monster)
                    bullet_hit = True
                    break

            if not bullet_hit:
                remaining_bullets.append(bullet)

        self.bullets = remaining_bullets
        self.monsters = [m for m in self.monsters if m not in killed_monsters]

        # Neue Wave, wenn alle Gegner tot sind.
        if len(self.monsters) == 0:
            self.wave += 1
            monster_count = self.curriculum_level + (self.wave - 1)
            for _ in range(monster_count):
                self.monsters.append(self.spawn_monster_on_edge())
            reward += self.wave_clear_reward

        # -------------------------
        # Monster bewegen
        # -------------------------
        for monster in self.monsters:
            monster.update(
                player=self.player,
                obstacles=self.obstacles,
                width=self.width,
                height=self.height,
                grid_map=self.grid_map,
            )

        player_hit = any(
            np.linalg.norm(monster.position - self.player.position) < 0.27
            for monster in self.monsters
        )

        # -------------------------
        # Wall / Obstacle Penalties
        # -------------------------
        margin = min(
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1],
        )

        if margin < 0.5:
            reward += self.wall_near_penalty

        obstacle_margin = min(
            obstacle.distance_obstacle_to_player(self.player.position[0], self.player.position[1])
            for obstacle in self.obstacles
        )

        if obstacle_margin < 0.3:
            reward += self.obstacle_near_penalty

        distances_to_walls = [
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1],
        ]
        distances_sorted = sorted(distances_to_walls)

        if distances_sorted[0] < 0.5 and distances_sorted[1] < 0.5:
            reward += self.corner_penalty

        if player_hit:
            reward += self.death_penalty
            terminated = True
        else:
            reward += self.alive_reward
            terminated = False

        truncated = self.current_step >= self.max_steps

        info = {
            "wave": self.wave,
            "monster_count": len(self.monsters),
            "bullet_count": len(self.bullets),
            "shoot_cooldown": self.shoot_cooldown,
            "last_shoot_dir": self.last_shoot_dir.copy(),
            "monster_states": [m.state.name for m in self.monsters],
            "monster_path_lengths": [len(m.path) for m in self.monsters],
            "monster_waypoint_indices": [m.current_waypoint_index for m in self.monsters],
            "monster_blocked_reasons": [
                getattr(m, "blocked_reason", None)
                for m in self.monsters
            ],
        }

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
