import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
import pygame
from enum import Enum, auto
from src.pathfinding import astar

# Player
class Player:
    def __init__(self, x, y, speed = 0.01, hp = 100):
        self.position = np.array([x, y], dtype=np.float32)
        self.speed = speed
        self.hp = hp

    def move(self, dx, dy, width, height):
        direction = np.array([dx, dy], dtype=np.float32)
        self.position += self.speed * direction
        self.position[0] = np.clip(self.position[0], 0, width)
        self.position[1] = np.clip(self.position[1], 0, height)


# Bullet
class Bullet:
    def __init__(self, x, y, dir_x, dir_y, speed = 0.3, damage = 50):
        self.position = np.array([x, y], dtype=np.float32)
        self.damage = damage
        self.speed = speed

        direction = np.array([dir_x, dir_y], dtype=np.float32)
        norm = np.linalg.norm(direction)
        self.direction = direction / norm if norm > 0 else np.array([1.0, 0.0])

    def update(self):
        self.position = self.position + self.speed * self.direction

    def out_of_bounds(self, width, height):
        return (self.position[0] < 0 or self.position[0] > width or
                self.position[1] < 0 or self.position[1] > height)

class MonsterState(Enum):
    DIRECT_CHASE = auto()
    PATHFINDING = auto()
    FOLLOW_PATH = auto()
    BLOCKED = auto()
    REACHED = auto()

# Monster
class Monster:
    def __init__(self, x, y, speed=0.005, hp=50):
        self.position = np.array([x, y], dtype=np.float32)
        self.speed = speed
        self.hp = hp

        self.state = MonsterState.DIRECT_CHASE

        # A*-Pfad als Weltkoordinaten-Wegpunkte
        self.path = []
        self.current_waypoint_index = 0

        # Merkt sich, für welches Ziel zuletzt geplant wurde
        self.last_goal_cell = None

        self.blocked_reason = None

    def update(self, player, obstacles, width, height, grid_map):
        """
        Zentrale Monsterlogik.

        Ablauf:
        1. Prüfen, ob Spieler erreicht wurde.
        2. Wenn direkte Sichtlinie frei ist: direkt verfolgen.
        3. Wenn direkte Sichtlinie blockiert ist: A*-Pfad planen.
        4. Wenn Pfad vorhanden ist: Pfad folgen.
        """

        if self._has_reached_player(player):
            self.state = MonsterState.REACHED
            return

        # Fall 1: Direkter Weg ist frei
        if self._has_line_of_sight(player, obstacles):
            self.path = []
            self.current_waypoint_index = 0
            self.last_goal_cell = None

            old_position = self.position.copy()
            self._move_directly_towards(player.position, width, height)

            if self._collides_with_obstacle(obstacles, radius=0.15):
                self.position = old_position
                self.state = MonsterState.BLOCKED
            else:
                self.state = MonsterState.DIRECT_CHASE
            return

        # Fall 2: Direkter Weg ist blockiert -> A* benutzen
        monster_cell = grid_map.world_to_grid(self.position)
        player_cell = grid_map.world_to_grid(player.position)

        needs_new_path = (
            not self.path
            or self.current_waypoint_index >= len(self.path)
            or self.last_goal_cell != player_cell
        )

        if needs_new_path:
            self.state = MonsterState.PATHFINDING
            self._plan_path(monster_cell, player_cell, grid_map)

        if self.path and self.current_waypoint_index < len(self.path):
            self.state = MonsterState.FOLLOW_PATH
            self._follow_path(obstacles, width, height)
        else:
            self.state = MonsterState.BLOCKED

    def _has_reached_player(self, player, kill_radius=0.27):
        return np.linalg.norm(self.position - player.position) < kill_radius

    def _move_directly_towards(self, target_position, width, height):
        direction = target_position - self.position
        norm = np.linalg.norm(direction)

        if norm > 0:
            direction = direction / norm

        self.position = self.position + self.speed * direction

        self.position[0] = np.clip(self.position[0], 0, width)
        self.position[1] = np.clip(self.position[1], 0, height)

    def _collides_with_obstacle(self, obstacles, radius=0.05):
        return any(
            obstacle.contains_p(
                self.position[0],
                self.position[1],
                radius=radius
            )
            for obstacle in obstacles
        )

    def _has_line_of_sight(self, player, obstacles, step_size=0.1, radius=0.15):
        """
        Prüft, ob die direkte Linie vom Monster zum Player frei ist.
        Dafür wird die Linie in kleine Punkte aufgeteilt.
        """

        start = self.position
        end = player.position

        direction = end - start
        distance = np.linalg.norm(direction)

        if distance == 0:
            return True

        direction = direction / distance
        steps = max(1, int(distance / step_size))

        for i in range(1, steps + 1):
            point = start + direction * step_size * i

            for obstacle in obstacles:
                if obstacle.contains_p(point[0], point[1], radius=radius):
                    return False

        return True

    def _plan_path(self, monster_cell, player_cell, grid_map):
        """
        Berechnet mit A* einen Pfad im Grid und wandelt ihn in Weltkoordinaten um.
        Falls Start- oder Zielzelle blockiert sind, wird die nächste freie Zelle gesucht.
        """

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

        if len(grid_path) > 1:
            grid_path = grid_path[1:]

        self.path = [
            grid_map.grid_to_world(cell)
            for cell in grid_path
        ]

        self.current_waypoint_index = 0
        self.last_goal_cell = player_cell
        self.blocked_reason = None

    def _follow_path(self, obstacles, width, height, waypoint_threshold=0.08):
        """
        Monster läuft zum nächsten Wegpunkt des berechneten Pfads.
        """

        if self.current_waypoint_index >= len(self.path):
            return

        waypoint = self.path[self.current_waypoint_index]

        old_position = self.position.copy()

        self._move_directly_towards(waypoint, width, height)

        if self._collides_with_obstacle(obstacles, radius=0.15):
            self.position = old_position
            self.path = []
            self.current_waypoint_index = 0
            self.state = MonsterState.BLOCKED
            self.blocked_reason = "collision_while_following_path"
            return

        distance_to_waypoint = np.linalg.norm(self.position - waypoint)

        if distance_to_waypoint < waypoint_threshold:
            self.current_waypoint_index += 1

    def move_toward(self, player):
        """
        Alte Methode bleibt erstmal erhalten.
        Später kann man sie löschen.
        """
        direction = np.array([
            player.position[0] - self.position[0],
            player.position[1] - self.position[1]
        ])

        norm = np.linalg.norm(direction)

        if norm > 0:
            direction = direction / norm

        self.position = self.position + self.speed * direction

# Obstacles einfügen
class Obstacle():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains_p(self, px, py, radius = 0.0):
        ''' Prüft ob Punkt Obstacle schneidet '''
        return (self.x - radius <= px <= self.x + self.width + radius and
                self.y - radius <= py <= self.y + self.height + radius)
    def distance_obstacle_to_player(self, px, py):
        dx = max(self.x - px, 0, px - (self.x + self.width))
        dy = max(self.y - py, 0, py - (self.y + self.height))
        return np.sqrt(dx**2 + dy**2)
