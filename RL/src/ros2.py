from __future__ import annotations

import json
import threading

import rclpy
from std_msgs.msg import Float32MultiArray, String

from src.environment import Environment


class TowerDefensePublisher:
    """
    Generic ROS2 wrapper for the environment.

    The action interface now expects four floats:
    [move_x, move_y, shoot_x, shoot_y].
    """

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("tower_defense_publisher")
        self.env = Environment()

        self.pub_game_state = self.node.create_publisher(
            Float32MultiArray,
            "/tower_defense/game_state",
            10,
        )
        self.pub_debug = self.node.create_publisher(
            String,
            "/tower_defense/debug",
            10,
        )

        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()
        self.node.get_logger().info("Tower Defense Publisher gestartet.")

    def _spin_ros(self) -> None:
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _get_env_state(self) -> dict[str, float]:
        state: dict[str, float] = {}

        if self.env.player is not None:
            state["player_x"] = float(self.env.player.position[0])
            state["player_y"] = float(self.env.player.position[1])

        state["monster_count"] = float(len(self.env.monsters))
        for i, monster in enumerate(self.env.monsters):
            state[f"m{i}_x"] = float(monster.position[0])
            state[f"m{i}_y"] = float(monster.position[1])
            state[f"m{i}_hp"] = float(monster.hp)

        for attr in [
            "wave",
            "curriculum_level",
            "current_step",
            "shoot_cooldown",
            "kills_this_episode",
            "hits_this_episode",
            "wave_clears_this_episode",
        ]:
            if hasattr(self.env, attr):
                state[attr] = float(getattr(self.env, attr))

        for i, bullet in enumerate(self.env.bullets[:4]):
            state[f"b{i}_x"] = float(bullet.position[0])
            state[f"b{i}_y"] = float(bullet.position[1])

        return state

    def publish_game_state(self) -> None:
        state = self._get_env_state()

        msg = Float32MultiArray()
        msg.data = list(state.values())
        self.pub_game_state.publish(msg)

        debug_msg = String()
        debug_msg.data = json.dumps(state)
        self.pub_debug.publish(debug_msg)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.publish_game_state()
        return obs, reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        self.publish_game_state()
        return obs, info

    def close(self) -> None:
        self.env.close()
        self.node.destroy_node()
