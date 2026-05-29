from __future__ import annotations

import numpy as np


class GridMap:
    def __init__(self, width: float, height: float, cell_size: float, obstacles, obstacle_margin: float = 0.15):
        self.width = float(width)
        self.height = float(height)
        self.cell_size = float(cell_size)
        self.obstacles = obstacles
        self.obstacle_margin = float(obstacle_margin)

        self.cols = int(np.ceil(self.width / self.cell_size))
        self.rows = int(np.ceil(self.height / self.cell_size))

    def world_to_grid(self, position) -> tuple[int, int]:
        x, y = position
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)

        gx = np.clip(gx, 0, self.cols - 1)
        gy = np.clip(gy, 0, self.rows - 1)
        return int(gx), int(gy)

    def grid_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        gx, gy = cell
        x = (gx + 0.5) * self.cell_size
        y = (gy + 0.5) * self.cell_size
        return np.array([x, y], dtype=np.float32)

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        gx, gy = cell
        return 0 <= gx < self.cols and 0 <= gy < self.rows

    def is_blocked(self, cell: tuple[int, int]) -> bool:
        if not self.in_bounds(cell):
            return True

        world_pos = self.grid_to_world(cell)
        return any(
            obstacle.contains_p(world_pos[0], world_pos[1], radius=self.obstacle_margin)
            for obstacle in self.obstacles
        )

    def find_nearest_free_cell(self, cell: tuple[int, int], max_radius: int = 8) -> tuple[int, int] | None:
        if not self.is_blocked(cell):
            return cell

        gx, gy = cell
        for radius in range(1, max_radius + 1):
            candidates: list[tuple[int, int]] = []

            for dx in range(-radius, radius + 1):
                candidates.append((gx + dx, gy - radius))
                candidates.append((gx + dx, gy + radius))

            for dy in range(-radius + 1, radius):
                candidates.append((gx - radius, gy + dy))
                candidates.append((gx + radius, gy + dy))

            for candidate in candidates:
                if self.in_bounds(candidate) and not self.is_blocked(candidate):
                    return candidate

        return None

    def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        gx, gy = cell
        candidates = [
            (gx + 1, gy),
            (gx - 1, gy),
            (gx, gy + 1),
            (gx, gy - 1),
        ]
        return [candidate for candidate in candidates if self.in_bounds(candidate) and not self.is_blocked(candidate)]
