"""
MuJoCo Visualizer für Tower Defense.

Verwendung:
    Terminal 1: python3 test_ros2.py
    Terminal 2: python3 mujoco_viz.py
"""

import threading
import json
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


MAX_MONSTERS = 16
MAX_BULLETS = 8


def make_monster_bodies(max_monsters=MAX_MONSTERS):
    return "\n".join(
        f'''
    <body name="monster_{i}" pos="0 0 -1">
      <freejoint name="monster_joint_{i}"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>'''
        for i in range(max_monsters)
    )


def make_bullet_bodies(max_bullets=MAX_BULLETS):
    return "\n".join(
        f'''
    <body name="bullet_{i}" pos="0 0 -1">
      <freejoint name="bullet_joint_{i}"/>
      <geom type="sphere" size="0.12" rgba="1 1 0 1"/>
    </body>'''
        for i in range(max_bullets)
    )


XML = f"""
<mujoco model="tower_defense">
  <option timestep="0.01"/>
  <worldbody>
    <light pos="3 3 10" dir="-1 -1 -2" diffuse="1 1 1"/>
    <camera name="topdown" pos="3 3 15" euler="0 0 0"/>
    <geom name="ground" type="plane" pos="3 3 0" size="3 3 0.01" rgba="0.2 0.2 0.2 1"/>

    <geom name="obstacle_0" type="box" pos="2.0 2.0 0.1" size="0.5 0.5 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="obstacle_1" type="box" pos="5.0 2.0 0.1" size="0.5 0.5 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="obstacle_2" type="box" pos="1.25 4.25 0.1" size="0.25 0.25 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="obstacle_3" type="box" pos="4.5 4.5 0.1" size="0.5 0.5 0.1" rgba="0.5 0.5 0.5 1"/>

    <body name="player" pos="3 3 0.1">
      <freejoint name="player_joint"/>
      <geom type="cylinder" size="0.15 0.05" rgba="1 1 1 1"/>
    </body>

    {make_monster_bodies()}

    {make_bullet_bodies()}
  </worldbody>
</mujoco>
"""


class TowerDefenseVisualizer(Node):
    def __init__(self, model, data):
        super().__init__("tower_defense_visualizer")

        self.model = model
        self.data = data

        self.player_pos = np.array([3.0, 3.0], dtype=np.float32)
        self.monster_positions = []
        self.bullet_positions = []

        self.create_subscription(
            String,
            "/tower_defense/debug",
            self._debug_callback,
            10,
        )

        self.player_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "player_joint",
        )

        self.monster_joint_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"monster_joint_{i}",
            )
            for i in range(MAX_MONSTERS)
        ]

        self.bullet_joint_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"bullet_joint_{i}",
            )
            for i in range(MAX_BULLETS)
        ]

        self.get_logger().info("MuJoCo Visualizer gestartet!")

    def _debug_callback(self, msg):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if "player_x" in state and "player_y" in state:
            self.player_pos = np.array(
                [state["player_x"], state["player_y"]],
                dtype=np.float32,
            )

        self.monster_positions = []
        for i in range(MAX_MONSTERS):
            if f"m{i}_x" in state and f"m{i}_y" in state:
                self.monster_positions.append(
                    np.array(
                        [state[f"m{i}_x"], state[f"m{i}_y"]],
                        dtype=np.float32,
                    )
                )

        self.bullet_positions = []
        for i in range(MAX_BULLETS):
            if f"b{i}_x" in state and f"b{i}_y" in state:
                self.bullet_positions.append(
                    np.array(
                        [state[f"b{i}_x"], state[f"b{i}_y"]],
                        dtype=np.float32,
                    )
                )

    def _set_freejoint_position(self, joint_id, x, y, z):
        if joint_id < 0:
            return

        addr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[addr + 0] = x
        self.data.qpos[addr + 1] = y
        self.data.qpos[addr + 2] = z
        self.data.qpos[addr + 3] = 1.0
        self.data.qpos[addr + 4] = 0.0
        self.data.qpos[addr + 5] = 0.0
        self.data.qpos[addr + 6] = 0.0

    def update_mujoco(self):
        self._set_freejoint_position(
            self.player_joint_id,
            self.player_pos[0],
            self.player_pos[1],
            0.1,
        )

        for i, joint_id in enumerate(self.monster_joint_ids):
            if i < len(self.monster_positions):
                self._set_freejoint_position(
                    joint_id,
                    self.monster_positions[i][0],
                    self.monster_positions[i][1],
                    0.1,
                )
            else:
                self._set_freejoint_position(joint_id, 0.0, 0.0, -1000.0)

        for i, joint_id in enumerate(self.bullet_joint_ids):
            if i < len(self.bullet_positions):
                self._set_freejoint_position(
                    joint_id,
                    self.bullet_positions[i][0],
                    self.bullet_positions[i][1],
                    0.1,
                )
            else:
                self._set_freejoint_position(joint_id, 0.0, 0.0, -1000.0)


def main():
    rclpy.init()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    visualizer = TowerDefenseVisualizer(model, data)

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(visualizer),
        daemon=True,
    )
    spin_thread.start()

    print("MuJoCo Visualizer läuft!")
    print("Starte in Terminal 1: python3 test_ros2.py")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            visualizer.update_mujoco()
            mujoco.mj_forward(model, data)
            viewer.sync()

    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
