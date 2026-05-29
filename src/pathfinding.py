from __future__ import annotations

import heapq


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    # Manhattan distance matches the 4-neighbor grid.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid_map, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    if grid_map.is_blocked(start) or grid_map.is_blocked(goal):
        return []

    open_set: list[tuple[int, int, tuple[int, int]]] = []
    tie_breaker = 0
    heapq.heappush(open_set, (0, tie_breaker, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}
    visited: set[tuple[int, int]] = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in grid_map.neighbors(current):
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                tie_breaker += 1
                heapq.heappush(open_set, (f_score, tie_breaker, neighbor))

    return []


def reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
