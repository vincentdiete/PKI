import numpy as np


class GridMap:
    def __init__(self, width, height, cell_size, obstacles, obstacle_margin=0.15):
        """
        Wandelt die kontinuierliche 2D-Welt in ein diskretes Planungsgrid um.

        width, height:
            Größe der Welt, bei euch 10 x 10.

        cell_size:
            Größe einer Grid-Zelle in Weltkoordinaten.
            Beispiel: cell_size = 0.25 -> 40 x 40 Grid.

        obstacles:
            Liste eurer Obstacle-Objekte.

        obstacle_margin:
            Künstliche Vergrößerung der Hindernisse.
            Wichtig, weil Monster nicht punktförmig sind.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.obstacles = obstacles
        self.obstacle_margin = obstacle_margin

        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))

    def world_to_grid(self, position):
        """
        Wandelt eine kontinuierliche Weltposition in eine Grid-Zelle um.

        Beispiel:
            position = [2.7, 4.1]
            cell_size = 0.25
            -> ungefähr (10, 16)
        """
        x, y = position

        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)

        gx = np.clip(gx, 0, self.cols - 1)
        gy = np.clip(gy, 0, self.rows - 1)

        return int(gx), int(gy)

    def grid_to_world(self, cell):
        """
        Wandelt eine Grid-Zelle zurück in Weltkoordinaten.
        Genommen wird der Mittelpunkt der Zelle.
        """
        gx, gy = cell

        x = (gx + 0.5) * self.cell_size
        y = (gy + 0.5) * self.cell_size

        return np.array([x, y], dtype=np.float32)

    def in_bounds(self, cell):
        """
        Prüft, ob eine Grid-Zelle innerhalb des Spielfelds liegt.
        """
        gx, gy = cell
        return 0 <= gx < self.cols and 0 <= gy < self.rows

    def is_blocked(self, cell):
        """
        Prüft, ob eine Grid-Zelle durch ein Hindernis blockiert ist.

        Vereinfachung:
        Wir prüfen den Mittelpunkt der Zelle.
        Wenn dieser in einem Hindernis liegt, gilt die Zelle als blockiert.
        """
        if not self.in_bounds(cell):
            return True

        world_pos = self.grid_to_world(cell)

        for obstacle in self.obstacles:
            if obstacle.contains_p(
                world_pos[0],
                world_pos[1],
                radius=self.obstacle_margin
            ):
                return True

        return False
    
    def find_nearest_free_cell(self, cell, max_radius=8):
        """
        Sucht um eine gegebene Zelle herum die nächste freie Zelle.
        Wird benutzt, falls Start- oder Zielzelle durch obstacle_margin blockiert ist.
        """
        if not self.is_blocked(cell):
            return cell

        gx, gy = cell

        for radius in range(1, max_radius + 1):
            candidates = []

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

    def neighbors(self, cell):
        """
        Gibt begehbare Nachbarzellen zurück.

        Erstmal nur 4 Richtungen:
        rechts, links, oben, unten.

        Keine Diagonalen, weil Monster sonst später eventuell
        unrealistisch durch Box-Ecken schneiden könnten.
        """
        gx, gy = cell

        candidates = [
            (gx + 1, gy),
            (gx - 1, gy),
            (gx, gy + 1),
            (gx, gy - 1),
        ]

        valid = []

        for candidate in candidates:
            if self.in_bounds(candidate) and not self.is_blocked(candidate):
                valid.append(candidate)

        return valid