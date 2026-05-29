import numpy as np

from src.entities import Player, Monster, Obstacle
from src.grid_map import GridMap


WIDTH = 10
HEIGHT = 10


def main():
    # Test-Situation:
    # Monster links, Player rechts, Box genau dazwischen.
    obstacles = [
        Obstacle(2.5, 2.5, 1.0, 1.0)
    ]

    grid_map = GridMap(
        width=WIDTH,
        height=HEIGHT,
        cell_size=0.25,
        obstacles=obstacles,
        obstacle_margin=0.15
    )

    player = Player(5.0, 3.0)
    monster = Monster(1.0, 3.0)

    # Nur für den Test schneller machen, sonst dauert es sehr lange.
    monster.speed = 0.03

    print("=== Startzustand ===")
    print("Player:", player.position)
    print("Monster:", monster.position)
    print("Obstacle: x=2.5..3.5, y=2.5..3.5")
    print()

    monster_cell = grid_map.world_to_grid(monster.position)
    player_cell = grid_map.world_to_grid(player.position)

    print("Monster grid cell:", monster_cell)
    print("Player grid cell:", player_cell)
    print("Monster cell blocked:", grid_map.is_blocked(monster_cell))
    print("Player cell blocked:", grid_map.is_blocked(player_cell))
    print()

    for step in range(500):
        monster.update(
            player=player,
            obstacles=obstacles,
            width=WIDTH,
            height=HEIGHT,
            grid_map=grid_map
        )

        state = monster.state.name
        dist = np.linalg.norm(monster.position - player.position)
        path_len = len(monster.path)
        waypoint_index = monster.current_waypoint_index

        collides = any(
            o.contains_p(
                monster.position[0],
                monster.position[1],
                radius=0.05
            )
            for o in obstacles
        )

        if step % 10 == 0 or state in ["BLOCKED", "REACHED"]:
            print(
                f"Step {step:03d} | "
                f"State: {state:12s} | "
                f"Pos: {monster.position} | "
                f"Dist: {dist:.3f} | "
                f"Path len: {path_len} | "
                f"Waypoint: {waypoint_index} | "
                f"Collision: {collides}"
            )

        if collides:
            print()
            print("FEHLER: Monster ist in ein Hindernis gelaufen.")
            break

        if state == "REACHED" or dist < 0.27:
            print()
            print("ERFOLG: Monster hat den Player erreicht.")
            break

    else:
        print()
        print("FEHLER: Monster hat den Player innerhalb von 500 Steps nicht erreicht.")


if __name__ == "__main__":
    main()