import heapq


def heuristic(a, b):
    """
    Manhattan-Distanz.
    Passt gut, weil wir im Grid erstmal nur 4 Nachbarn benutzen:
    rechts, links, oben, unten.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid_map, start, goal):
    """
    Berechnet einen Pfad von start nach goal auf der GridMap.

    start, goal:
        Grid-Zellen, z. B. (3, 8)

    Rückgabe:
        Liste von Grid-Zellen, z. B.
        [(3, 8), (4, 8), (5, 8), ...]
        oder [] falls kein Pfad gefunden wurde.
    """

    if grid_map.is_blocked(start):
        return []

    if grid_map.is_blocked(goal):
        return []

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {
        start: 0
    }

    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

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
                heapq.heappush(open_set, (f_score, neighbor))

    return []


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path