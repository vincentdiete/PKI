# Libraries
import numpy as np


# Player (Initialentwurf)
class Player:

    def __init__(self, x, y, speed = 1.0, hp = 100):
        self.position = np.array([x,y], dtype = np.float32)
        self.speed = speed
        self.hp = hp

    def move(self, dx, dy, width, height):
        direction = np.array([dx, dy], dtype = np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        self.position += self.speed * direction
        self.position[0] = np.clip(self.position[0], 0 ,width)
        self.position[1] = np.clip(self.position[1], 0, height)

# Monster (Initialentwurf)
class Monster:

    def __init__(self, x, y, speed = 1.0, hp = 100):
        self.position = np.array([x,y], dtype = np.float32)
        self.speed = speed
        self.hp = hp

    def move_toward(self, player):
        direction = np.array([player.position[0] - self.position[0], player.position[1] - self.position[1]])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        self.position = self.position + self.speed * direction


