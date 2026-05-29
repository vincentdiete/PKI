from __future__ import annotations

from enum import Enum, auto
from typing import Iterable, Optional

import numpy as np

from src.pathfinding import astar


class Player:
    def __init__(self, x: float, y: float, speed: float = 0.01, hp: int = 100):
        self.position = np.array([x, y], dtype=np.float32)
        self.speed = float(speed)
        self.hp = int(hp)

    def move(self, dx: float, dy: float, width: float, height: float) -> None:
        """Move by an already-normalized direction vector and clamp to the world."""
        direction = np.array([dx, dy], dtype=np.float32)
        self.position = self.position + self.speed * direction
        self.position[0] = np.clip(self.position[0], 0.0, width)
        self.position[1] = np.clip(self.position[1], 0.0, height)


class Bullet:
    def __init__(
        self,
        x: float,
        y: float,
        dir_x: float,
        dir_y: float,
        speed: float = 0.3,
        damage: int = 50,
    ):
        self.position = np.array([x, y], dtype=np.float32)
        self.damage = int(damage)
        self.speed = float(speed)

        direction = np.array([dir_x, dir_y], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        self.direction = direction / norm if norm > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)

    def update(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance the bullet and return (old_position, new_position).
        The returned segment is used for robust hit detection at high bullet speeds.
        """
        previous_position = self.position.copy()
        self.position = self.position + self.speed * self.direction
        return previous_position, self.position.copy()

    def out_of_bounds(self, width: float, height: float) -> bool:
        return (
            self.position[0] < 0.0
            or self.position[0] > width
            or self.position[1] < 0.0
            or self.position[1] > height
        )


class MonsterState(Enum):
    DIRECT_CHASE = auto()
    PATHFINDING = auto()
    FOLLOW_PATH = auto()
    BLOCKED = auto()
    REACHED = auto()


class Monster:
    def __init__(self, x: float, y: float, speed: float = 0.005, hp: int = 100):
        self.position = np.array([x, y], dtype=np.float32)
        self.speed = float(speed)
        self.hp = int(hp)
        self.max_hp = int(hp)

        self.state = MonsterState.DIRECT_CHASE
        self.path: list[np.ndarray] = []
        self.current_waypoint_index = 0
        self.last_goal_cell: Optional[tuple[int, int]] = None
        self.steps_since_replan = 0
        self.blocked_reason: Optional[str] = None

    def update(self, player: Player, obstacles: Iterable["Obstacle"], width: float, height: float, grid_map) -> None:
        """
        Monster behavior:
        1. If player was reached, mark REACHED.
        2. If line of sight is free, chase directly.
        3. Otherwise use A* and follow waypoints.

        Replanning is deliberately throttled. Replanning on every tiny player movement
        creates unnecessary non-stationarity and CPU overhead without making the enemy smarter.
        """
        self.steps_since_replan += 1

        if self._has_reached_player(player):
            self.state = MonsterState.REACHED
            return

        if self._has_line_of_sight(player, obstacles):
            self.path = []
            self.current_waypoint_index = 0
            self.last_goal_cell = None
            self.steps_since_replan = 0
            self.blocked_reason = None

            old_position = self.position.copy()
            self._move_directly_towards(player.position, width, height)

            if self._collides_with_obstacle(obstacles, radius=0.05):
                self.position = old_position
                self.state = MonsterState.BLOCKED
                self.blocked_reason = "direct_chase_collision"
            else:
                self.state = MonsterState.DIRECT_CHASE
            return

        monster_cell = grid_map.world_to_grid(self.position)
        player_cell = grid_map.world_to_grid(player.position)

        needs_new_path = (
            not self.path
            or self.current_waypoint_index >= len(self.path)
            or (
                self.last_goal_cell != player_cell
                and self.steps_since_replan >= 20
            )
        )

        if needs_new_path:
            self.state = MonsterState.PATHFINDING
            self._plan_path(monster_cell, player_cell, grid_map)
            self.steps_since_replan = 0

        if self.path and self.current_waypoint_index < len(self.path):
            self.state = MonsterState.FOLLOW_PATH
            self._follow_path(obstacles, width, height)
        else:
            self.state = MonsterState.BLOCKED
            if self.blocked_reason is None:
                self.blocked_reason = "no_active_path"

    def _has_reached_player(self, player: Player, kill_radius: float = 0.27) -> bool:
        return float(np.linalg.norm(self.position - player.position)) < kill_radius

    def _move_directly_towards(self, target_position: np.ndarray, width: float, height: float) -> None:
        direction = target_position - self.position
        norm = float(np.linalg.norm(direction))

        if norm > 1e-8:
            direction = direction / norm
        else:
            direction = np.zeros(2, dtype=np.float32)

        self.position = self.position + self.speed * direction
        self.position[0] = np.clip(self.position[0], 0.0, width)
        self.position[1] = np.clip(self.position[1], 0.0, height)

    def _collides_with_obstacle(self, obstacles: Iterable["Obstacle"], radius: float = 0.05) -> bool:
        return any(
            obstacle.contains_p(self.position[0], self.position[1], radius=radius)
            for obstacle in obstacles
        )

    def _has_line_of_sight(
        self,
        player: Player,
        obstacles: Iterable["Obstacle"],
        step_size: float = 0.1,
        radius: float = 0.05,
    ) -> bool:
        start = self.position
        end = player.position
        direction = end - start
        distance = float(np.linalg.norm(direction))

        if distance <= 1e-8:
            return True

        direction = direction / distance
        steps = max(1, int(distance / step_size))

        for i in range(1, steps + 1):
            point = start + direction * min(step_size * i, distance)
            if any(obstacle.contains_p(point[0], point[1], radius=radius) for obstacle in obstacles):
                return False

        return True

    def _plan_path(self, monster_cell: tuple[int, int], player_cell: tuple[int, int], grid_map) -> None:
        start_cell = grid_map.find_nearest_free_cell(monster_cell)
        goal_cell = grid_map.find_nearest_free_cell(player_cell)

        if start_cell is None or goal_cell is None:
            self.path = []
            self.current_waypoint_index = 0
            self.last_goal_cell = player_cell
            self.blocked_reason = "no_free_start_or_goal_cell"
            return

        grid_path = astar(grid_map, start_cell, goal_cell)

        if not grid_path:
            self.path = []
            self.current_waypoint_index = 0
            self.last_goal_cell = player_cell
            self.blocked_reason = "astar_no_path"
            return

        # Skip current cell; otherwise the monster may waste steps walking to its own cell center.
        if len(grid_path) > 1:
            grid_path = grid_path[1:]

        self.path = [grid_map.grid_to_world(cell) for cell in grid_path]
        self.current_waypoint_index = 0
        self.last_goal_cell = player_cell
        self.blocked_reason = None

    def _follow_path(self, obstacles: Iterable["Obstacle"], width: float, height: float, waypoint_threshold: float = 0.08) -> None:
        if self.current_waypoint_index >= len(self.path):
            return

        waypoint = self.path[self.current_waypoint_index]
        old_position = self.position.copy()
        self._move_directly_towards(waypoint, width, height)

        if self._collides_with_obstacle(obstacles, radius=0.05):
            self.position = old_position
            self.path = []
            self.current_waypoint_index = 0
            self.state = MonsterState.BLOCKED
            self.blocked_reason = "collision_while_following_path"
            return

        if float(np.linalg.norm(self.position - waypoint)) < waypoint_threshold:
            self.current_waypoint_index += 1


class Goblin(Monster):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, speed=0.0065, hp=50)


class Golem(Monster):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, speed=0.0035, hp=150)


class Obstacle:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    def contains_p(self, px: float, py: float, radius: float = 0.0) -> bool:
        return (
            self.x - radius <= px <= self.x + self.width + radius
            and self.y - radius <= py <= self.y + self.height + radius
        )

    def distance_to_point(self, px: float, py: float) -> float:
        dx = max(self.x - px, 0.0, px - (self.x + self.width))
        dy = max(self.y - py, 0.0, py - (self.y + self.height))
        return float(np.sqrt(dx**2 + dy**2))

    def distance_obstacle_to_player(self, px: float, py: float) -> float:
        # Backwards-compatible name used by older code.
        return self.distance_to_point(px, py)
