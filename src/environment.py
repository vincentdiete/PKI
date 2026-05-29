from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.entities import Bullet, Goblin, Golem, Monster, Obstacle, Player
from src.grid_map import GridMap


class Environment(gym.Env):
    """
    2D tower-defense/shooter environment for SAC.

    Main RL-oriented changes compared with the earlier version:
    - vector action space instead of angle + trigger,
    - normalized observations with realistic bounds,
    - direct hit and kill rewards,
    - capped episode length,
    - curriculum based on wave-clears/kills instead of pure survival time,
    - robust bullet segment collision.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.width = 10.0
        self.height = 10.0
        self.max_distance = float(np.sqrt(self.width**2 + self.height**2))

        self.max_monsters_observed = 4
        self.max_obstacles_observed = 2
        self.max_monster_hp = 150.0
        self.max_wave_for_obs = 10.0
        self.max_curriculum_level = 4
        self.max_monsters_alive = self.max_monsters_observed

        self.obstacles = [
            Obstacle(2.5, 2.5, 1.0, 1.0),
            Obstacle(7.5, 2.5, 1.0, 1.0),
            Obstacle(1.5, 6.0, 0.5, 0.5),
            Obstacle(7.0, 7.0, 1.0, 1.0),
        ]

        self.grid_map = GridMap(
            width=self.width,
            height=self.height,
            cell_size=0.25,
            obstacles=self.obstacles,
            obstacle_margin=0.15,
        )

        # Curriculum and episode limits.
        self.curriculum_level = 1
        self.curriculum_history: list[dict[str, float]] = []
        self.curriculum_window = 20
        self.min_avg_wave_clears_for_level_up = 0.8
        self.min_avg_kills_for_level_up = 1.0
        self.max_steps = 3000

        # Reward constants. Keep these centralized for controlled experiments.
        self.survival_reward = 0.01
        self.hit_reward = 1.0
        self.kill_reward = 4.0
        self.wave_clear_reward = 8.0
        self.death_penalty = -10.0
        self.wall_near_penalty = -0.2
        self.corner_penalty = -0.3
        self.obstacle_collision_penalty = -0.1

        # Control constants.
        self.move_deadzone = 0.10
        self.shoot_deadzone = 0.35
        self.shoot_cooldown_steps = 30
        self.bullet_hit_radius = 0.20

        # New action space: [move_x, move_y, shoot_x, shoot_y].
        # Small move vector -> stand still.
        # Small shoot vector -> do not shoot.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation layout:
        # player position norm:                     2
        # wall distances norm:                      4
        # cooldown norm:                            1
        # nearest monsters: 4 * [present, dx, dy, dist, hp] = 20
        # nearest obstacles: 2 * [dx, dy, surface_dist, near_flag] = 8
        # counters: [monster_count, wave, curriculum_level] = 3
        # total: 38
        obs_low = [0.0, 0.0]                      # player position
        obs_high = [1.0, 1.0]

        obs_low += [0.0] * 4                      # walls
        obs_high += [1.0] * 4

        obs_low += [0.0]                          # cooldown
        obs_high += [1.0]

        for _ in range(self.max_monsters_observed):
            obs_low += [0.0, -1.0, -1.0, 0.0, 0.0]
            obs_high += [1.0, 1.0, 1.0, 1.0, 1.0]

        for _ in range(self.max_obstacles_observed):
            obs_low += [-1.0, -1.0, 0.0, 0.0]
            obs_high += [1.0, 1.0, 1.0, 1.0]

        obs_low += [0.0, 0.0, 0.0]                # counters
        obs_high += [1.0, 1.0, 1.0]

        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.player: Player | None = None
        self.monsters: list[Monster] = []
        self.bullets: list[Bullet] = []
        self.shoot_cooldown = 0
        self.wave = 0
        self.current_step = 0
        self.kills_this_episode = 0
        self.hits_this_episode = 0
        self.wave_clears_this_episode = 0
        self.last_position = None

    def _get_obs(self) -> np.ndarray:
        assert self.player is not None

        obs: list[float] = []

        # Player position and wall distances, normalized to [0, 1].
        px, py = self.player.position
        obs.extend([float(px / self.width), float(py / self.height)])

        dist_left = float(px / self.width)
        dist_right = float((self.width - px) / self.width)
        dist_down = float(py / self.height)
        dist_up = float((self.height - py) / self.height)
        obs.extend([dist_left, dist_right, dist_down, dist_up])

        obs.append(float(self.shoot_cooldown / self.shoot_cooldown_steps))

        # Nearest monsters. Relative positions avoid angle discontinuities.
        sorted_monsters = sorted(self.monsters, key=lambda m: np.linalg.norm(m.position - self.player.position))
        for i in range(self.max_monsters_observed):
            if i < len(sorted_monsters):
                monster = sorted_monsters[i]
                delta = monster.position - self.player.position
                dist = float(np.linalg.norm(delta))
                obs.extend([
                    1.0,
                    float(np.clip(delta[0] / self.width, -1.0, 1.0)),
                    float(np.clip(delta[1] / self.height, -1.0, 1.0)),
                    float(np.clip(dist / self.max_distance, 0.0, 1.0)),
                    float(np.clip(monster.hp / self.max_monster_hp, 0.0, 1.0)),
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Nearest obstacles by surface distance. No punishment for being close is encoded here;
        # the policy merely receives useful geometry.
        sorted_obstacles = sorted(
            self.obstacles,
            key=lambda o: o.distance_to_point(self.player.position[0], self.player.position[1]),
        )
        for i in range(self.max_obstacles_observed):
            if i < len(sorted_obstacles):
                obstacle = sorted_obstacles[i]
                center_x = obstacle.x + obstacle.width / 2.0
                center_y = obstacle.y + obstacle.height / 2.0
                delta_x = center_x - self.player.position[0]
                delta_y = center_y - self.player.position[1]
                surface_dist = obstacle.distance_to_point(self.player.position[0], self.player.position[1])
                near_flag = 1.0 if surface_dist < 0.3 else 0.0
                obs.extend([
                    float(np.clip(delta_x / self.width, -1.0, 1.0)),
                    float(np.clip(delta_y / self.height, -1.0, 1.0)),
                    float(np.clip(surface_dist / self.max_distance, 0.0, 1.0)),
                    near_flag,
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        obs.extend([
            float(np.clip(len(self.monsters) / self.max_monsters_alive, 0.0, 1.0)),
            float(np.clip(self.wave / self.max_wave_for_obs, 0.0, 1.0)),
            float(np.clip(self.curriculum_level / self.max_curriculum_level, 0.0, 1.0)),
        ])

        obs_array = np.array(obs, dtype=np.float32)
        return np.clip(obs_array, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._maybe_update_curriculum_from_finished_episode()

        self.current_step = 0
        self.wave = 0
        self.shoot_cooldown = 0
        self.bullets = []
        self.last_position = None
        self.kills_this_episode = 0
        self.hits_this_episode = 0
        self.wave_clears_this_episode = 0

        self.player = self._spawn_player()
        self.monsters = [self.spawn_monster_on_edge() for _ in range(self.curriculum_level)]

        return self._get_obs(), {}

    def _maybe_update_curriculum_from_finished_episode(self) -> None:
        if self.current_step <= 0:
            return

        self.curriculum_history.append({
            "steps": float(self.current_step),
            "kills": float(self.kills_this_episode),
            "hits": float(self.hits_this_episode),
            "wave_clears": float(self.wave_clears_this_episode),
        })
        self.curriculum_history = self.curriculum_history[-self.curriculum_window:]

        if len(self.curriculum_history) < self.curriculum_window:
            return

        avg_wave_clears = float(np.mean([h["wave_clears"] for h in self.curriculum_history]))
        avg_kills = float(np.mean([h["kills"] for h in self.curriculum_history]))

        if (
            self.curriculum_level < self.max_curriculum_level
            and avg_wave_clears >= self.min_avg_wave_clears_for_level_up
            and avg_kills >= self.min_avg_kills_for_level_up
        ):
            self.curriculum_level += 1
            self.curriculum_history = []

    def _spawn_player(self) -> Player:
        margin = 3.5
        for _ in range(1000):
            px = float(self.np_random.uniform(margin, self.width - margin))
            py = float(self.np_random.uniform(margin, self.height - margin))
            if not any(obstacle.contains_p(px, py, radius=0.2) for obstacle in self.obstacles):
                return Player(px, py)
        # Fallback should practically never be reached with the current map.
        return Player(self.width / 2.0, self.height / 2.0)

    def spawn_monster_on_edge(self) -> Monster:
        side = int(self.np_random.integers(0, 4))
        if side == 0:
            x, y = float(self.np_random.uniform(0.0, self.width)), 0.0
        elif side == 1:
            x, y = float(self.np_random.uniform(0.0, self.width)), self.height
        elif side == 2:
            x, y = 0.0, float(self.np_random.uniform(0.0, self.height))
        else:
            x, y = self.width, float(self.np_random.uniform(0.0, self.height))

        # Keep the default Monster dominant for stable early learning.
        # Introduce type variation only after the agent has learned the basic task.
        if self.curriculum_level >= 3:
            roll = float(self.np_random.random())
            if roll < 0.20:
                return Goblin(x, y)
            if roll > 0.85:
                return Golem(x, y)
        return Monster(x, y)

    def step(self, action):
        assert self.player is not None

        self.current_step += 1
        reward = 0.0

        action = np.asarray(action, dtype=np.float32)
        move_vec = self._normalize_or_zero(action[:2], self.move_deadzone)
        shoot_vec = self._normalize_or_zero(action[2:], self.shoot_deadzone)

        reward += self._apply_player_movement(move_vec)
        self._maybe_shoot(shoot_vec)
        self._tick_cooldown()

        bullet_reward = self._update_bullets_and_apply_hits()
        reward += bullet_reward

        if len(self.monsters) == 0:
            reward += self._advance_wave()

        for monster in self.monsters:
            monster.update(
                player=self.player,
                obstacles=self.obstacles,
                width=self.width,
                height=self.height,
                grid_map=self.grid_map,
            )

        terminated = self._player_was_caught()
        if terminated:
            reward += self.death_penalty
        else:
            reward += self.survival_reward
            reward += self._small_boundary_penalty()

        truncated = self.current_step >= self.max_steps
        obs = self._get_obs()

        info = {
            "wave": self.wave,
            "curriculum_level": self.curriculum_level,
            "kills": self.kills_this_episode,
            "hits": self.hits_this_episode,
            "wave_clears": self.wave_clears_this_episode,
            "active_monsters": len(self.monsters),
            "shoot_cooldown": self.shoot_cooldown,
            "monster_states": [m.state.name for m in self.monsters],
            "monster_path_lengths": [len(m.path) for m in self.monsters],
            "monster_waypoint_indices": [m.current_waypoint_index for m in self.monsters],
            "monster_blocked_reasons": [m.blocked_reason for m in self.monsters],
            "is_success": self.wave_clears_this_episode > 0,
        }

        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _normalize_or_zero(vector: np.ndarray, deadzone: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < deadzone:
            return np.zeros(2, dtype=np.float32)
        return (vector / max(norm, 1e-8)).astype(np.float32)

    def _apply_player_movement(self, move_vec: np.ndarray) -> float:
        assert self.player is not None
        if float(np.linalg.norm(move_vec)) <= 1e-8:
            return 0.0

        previous_position = self.player.position.copy()
        self.player.move(move_vec[0], move_vec[1], self.width, self.height)

        if any(obstacle.contains_p(self.player.position[0], self.player.position[1], radius=0.05) for obstacle in self.obstacles):
            self.player.position = previous_position
            return self.obstacle_collision_penalty

        return 0.0

    def _maybe_shoot(self, shoot_vec: np.ndarray) -> None:
        assert self.player is not None
        if self.shoot_cooldown != 0:
            return
        if float(np.linalg.norm(shoot_vec)) <= 1e-8:
            return

        self.bullets.append(
            Bullet(
                self.player.position[0],
                self.player.position[1],
                shoot_vec[0],
                shoot_vec[1],
            )
        )
        self.shoot_cooldown = self.shoot_cooldown_steps

    def _tick_cooldown(self) -> None:
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def _update_bullets_and_apply_hits(self) -> float:
        reward = 0.0
        remaining_bullets: list[Bullet] = []

        for bullet in self.bullets:
            old_pos, new_pos = bullet.update()

            if bullet.out_of_bounds(self.width, self.height):
                continue

            if self._segment_hits_any_obstacle(old_pos, new_pos):
                continue

            hit_monster = self._find_first_monster_hit_by_segment(old_pos, new_pos)
            if hit_monster is None:
                remaining_bullets.append(bullet)
                continue

            hit_monster.hp -= bullet.damage
            self.hits_this_episode += 1
            reward += self.hit_reward

            if hit_monster.hp <= 0:
                self.kills_this_episode += 1
                reward += self.kill_reward

        self.bullets = remaining_bullets
        self.monsters = [monster for monster in self.monsters if monster.hp > 0]
        return reward

    def _find_first_monster_hit_by_segment(self, start: np.ndarray, end: np.ndarray) -> Monster | None:
        candidates = []
        for monster in self.monsters:
            if monster.hp <= 0:
                continue
            distance = self._point_to_segment_distance(monster.position, start, end)
            if distance <= self.bullet_hit_radius:
                # Use distance along segment for deterministic nearest-hit behavior.
                along = self._segment_projection_factor(monster.position, start, end)
                candidates.append((along, monster))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _segment_hits_any_obstacle(self, start: np.ndarray, end: np.ndarray) -> bool:
        # Cheap and robust enough for short bullet segments: sample along the segment.
        segment_length = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(segment_length / 0.05)))
        for i in range(sample_count + 1):
            alpha = i / sample_count
            point = start + alpha * (end - start)
            if any(obstacle.contains_p(point[0], point[1]) for obstacle in self.obstacles):
                return True
        return False

    @staticmethod
    def _segment_projection_factor(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(point - start, segment) / segment_len_sq, 0.0, 1.0))

    @classmethod
    def _point_to_segment_distance(cls, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        alpha = cls._segment_projection_factor(point, start, end)
        closest = start + alpha * (end - start)
        return float(np.linalg.norm(point - closest))

    def _advance_wave(self) -> float:
        self.wave += 1
        self.wave_clears_this_episode += 1

        monster_count = min(self.max_monsters_alive, self.curriculum_level + self.wave - 1)
        self.monsters = [self.spawn_monster_on_edge() for _ in range(monster_count)]
        return self.wave_clear_reward

    def _player_was_caught(self) -> bool:
        assert self.player is not None
        return any(float(np.linalg.norm(monster.position - self.player.position)) < 0.27 for monster in self.monsters)

    def _small_boundary_penalty(self) -> float:
        assert self.player is not None
        margin = min(
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1],
        )

        penalty = 0.0
        if margin < 0.25:
            penalty += self.wall_near_penalty

        distances = [
            self.player.position[0],
            self.width - self.player.position[0],
            self.player.position[1],
            self.height - self.player.position[1],
        ]
        distances_sorted = sorted(float(d) for d in distances)
        if distances_sorted[0] < 0.25 and distances_sorted[1] < 0.25:
            penalty += self.corner_penalty

        return penalty

    def get_frame(self) -> np.ndarray:
        """Simple RGB frame for debugging. Not used during training."""
        scale = 20
        frame = np.zeros((int(self.height * scale), int(self.width * scale), 3), dtype=np.uint8)

        if self.player is not None:
            self._draw_circle(frame, self.player.position, radius=4, color=(255, 255, 255), scale=scale)

        for monster in self.monsters:
            self._draw_circle(frame, monster.position, radius=5, color=(255, 0, 0), scale=scale)

        for bullet in self.bullets:
            self._draw_circle(frame, bullet.position, radius=2, color=(255, 255, 0), scale=scale)

        for obstacle in self.obstacles:
            x0 = int(obstacle.x * scale)
            y0 = int(obstacle.y * scale)
            x1 = int((obstacle.x + obstacle.width) * scale)
            y1 = int((obstacle.y + obstacle.height) * scale)
            frame[y0:y1, x0:x1] = (80, 80, 80)

        return frame

    @staticmethod
    def _draw_circle(frame: np.ndarray, position: np.ndarray, radius: int, color: tuple[int, int, int], scale: int) -> None:
        cx = int(position[0] * scale)
        cy = int(position[1] * scale)
        height, width = frame.shape[:2]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        frame[y, x] = color

    def close(self) -> None:
        self.bullets = []
        self.monsters = []
