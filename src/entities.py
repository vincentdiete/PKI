import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
import pygame

# Player
class Player:
    def __init__(self, x, y, speed = 0.75, hp = 100):
        self.position = np.array([x, y], dtype=np.float32)
        self.speed = speed
        self.hp = hp

    def move(self, dx, dy, width, height):
        direction = np.array([dx, dy], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        self.position += self.speed * direction
        self.position[0] = np.clip(self.position[0], 0, width)
        self.position[1] = np.clip(self.position[1], 0, height)


# Bullet
class Bullet:
    def __init__(self, x, y, dir_x, dir_y, speed = 2.5, damage = 50):
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


# Monster
class Monster:
    def __init__(self, x, y, speed = 0.45, hp = 100):
        self.position = np.array([x, y], dtype=np.float32)
        self.speed = speed
        self.hp = hp

    def move_toward(self, player):
        direction = np.array([
            player.position[0] - self.position[0],
            player.position[1] - self.position[1]
        ])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        self.position = self.position + self.speed * direction

class Goblin(Monster):
    # Schnelles, schwaches Monster (oneshot)
    def __init__(self, x, y):
        super().__init__(x, y, speed = 0.65, hp = 50)

class Golem(Monster):
    # Langsames, starkes Monster (3 Treffer nötig bei dmg = 50)
    def __init__(self, x, y):
        super().__init__(x, y, speed = 0.25, hp = 150)