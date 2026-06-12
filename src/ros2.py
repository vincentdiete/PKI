import threading
import json
import numpy as np
import rclpy
from std_msgs.msg import Float32MultiArray, String

from src.combat_env import CombatEnv


class TowerDefensePublisher:
    """
    ROS2-Wrapper für die kombinierte CombatEnv.

    Erwartung:
    - env.step(action) bekommt 4D-Action: [move_x, move_y, shoot_x, shoot_y]
    - Movement-Observation wird an den Movement-Agenten zurückgegeben.
    - Shooting-Observation kann über publisher.env.get_shooting_obs() geholt werden.
    """

    def __init__(self, env=None):
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("tower_defense_publisher")
        self.env = env if env is not None else CombatEnv()

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
        self.node.get_logger().info("Tower Defense Publisher gestartet!")

    def _spin_ros(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _get_env_state(self):
        state = {}

        if self.env.player is not None:
            state["player_x"] = float(self.env.player.position[0])
            state["player_y"] = float(self.env.player.position[1])

        state["monster_count"] = float(len(self.env.monsters))
        for i, monster in enumerate(self.env.monsters):
            state[f"m{i}_x"] = float(monster.position[0])
            state[f"m{i}_y"] = float(monster.position[1])
            state[f"m{i}_hp"] = float(getattr(monster, "hp", 0.0))

        for attr in ["wave", "curriculum_level", "current_step", "shoot_cooldown"]:
            if hasattr(self.env, attr):
                state[attr] = float(getattr(self.env, attr))

        if hasattr(self.env, "last_shoot_dir"):
            state["shoot_x"] = float(self.env.last_shoot_dir[0])
            state["shoot_y"] = float(self.env.last_shoot_dir[1])
        else:
            state["shoot_x"] = 0.0
            state["shoot_y"] = 0.0

        for i, bullet in enumerate(self.env.bullets):
            state[f"b{i}_x"] = float(bullet.position[0])
            state[f"b{i}_y"] = float(bullet.position[1])

        return state

    def publish_game_state(self):
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

    def close(self):
        self.node.destroy_node()
