import threading
import rclpy
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import json
from src.environment import Environment

class TowerDefensePublisher:
    """
    Generischer Wrapper - passt sich automatisch an Environment Änderungen an.
    """

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node('tower_defense_publisher')
        self.env = Environment()

        
        self.pub_game_state = self.node.create_publisher(
            Float32MultiArray,
            '/tower_defense/game_state',
            10
        )

        self.pub_debug = self.node.create_publisher(
            String,
            '/tower_defense/debug',
            10
        )

        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()
        self.node.get_logger().info('Tower Defense Publisher gestartet!')

    def _spin_ros(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _get_env_state(self):
        state = {}
        
        
        if self.env.player is not None:
            state['player_x'] = float(self.env.player.position[0])
            state['player_y'] = float(self.env.player.position[1])
        
        
        state['monster_count'] = len(self.env.monsters)
        for i, m in enumerate(self.env.monsters):
            state[f'm{i}_x'] = float(m.position[0])
            state[f'm{i}_y'] = float(m.position[1])
            state[f'm{i}_hp'] = float(m.hp)
        
        for attr in ['wave', 'curriculum_level', 'current_step', 'shoot_cooldown']:
            if hasattr(self.env, attr):
                state[attr] = float(getattr(self.env, attr))

        if hasattr(self.env, "last_shoot_dir"):
            state["shoot_x"] = float(self.env.last_shoot_dir[0])
            state["shoot_y"] = float(self.env.last_shoot_dir[1])
        else:
            state["shoot_x"] = 0.0
            state["shoot_y"] = 0.0
        

        for i, b in enumerate(self.env.bullets[:4]):
            state[f'b{i}_x'] = float(b.position[0])
            state[f'b{i}_y'] = float(b.position[1])
        
        return state
    def publish_game_state(self):
        """Publisht alles - passt sich automatisch an."""
        state = self._get_env_state()

        # Als Float Array (für RL Agent)
        data = list(state.values())
        msg = Float32MultiArray()
        msg.data = data
        self.pub_game_state.publish(msg)

        # Als JSON String (für Debugging)
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
