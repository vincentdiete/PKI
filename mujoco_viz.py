"""
MuJoCo Visualizer für Tower Defense.
Subscribt ROS2 Topics und zeigt Spielzustand in MuJoCo.
 
Verwendung:
    Terminal 1: python3 test_ros2.py      (Publisher + Training)
    Terminal 2: python3 mujoco_viz.py     (MuJoCo Fenster)
"""
 
import threading
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import json
 
MAX_MONSTERS = 16
 
XML = """
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

    <!-- Spieler (weiß) -->
    <body name="player" pos="3 3 0.1">
      <freejoint name="player_joint"/>
      <geom type="cylinder" size="0.15 0.05" rgba="1 1 1 1"/>
    </body>
 
    <!-- Monster (rot) -->
    <body name="monster_0" pos="0 0 -1">
      <freejoint name="monster_joint_0"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_1" pos="0 0 -1">
      <freejoint name="monster_joint_1"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_2" pos="0 0 -1">
      <freejoint name="monster_joint_2"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_3" pos="0 0 -1">
      <freejoint name="monster_joint_3"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_4" pos="0 0 -1">
      <freejoint name="monster_joint_4"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_5" pos="0 0 -1">
      <freejoint name="monster_joint_5"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_6" pos="0 0 -1">
      <freejoint name="monster_joint_6"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
    <body name="monster_7" pos="0 0 -1">
      <freejoint name="monster_joint_7"/>
      <geom type="cylinder" size="0.25 0.05" rgba="1 0 0 1"/>
    </body>
 
    <!-- Bullets (gelb) -->
    <body name="bullet_0" pos="0 0 -1">
      <freejoint name="bullet_joint_0"/>
      <geom type="sphere" size="0.12" rgba="1 1 0 1"/>
    </body>
    <body name="bullet_1" pos="0 0 -1">
      <freejoint name="bullet_joint_1"/>
      <geom type="sphere" size="0.12" rgba="1 1 0 1"/>
    </body>
    <body name="bullet_2" pos="0 0 -1">
      <freejoint name="bullet_joint_2"/>
      <geom type="sphere" size="0.12" rgba="1 1 0 1"/>
    </body>
    <body name="bullet_3" pos="0 0 -1">
      <freejoint name="bullet_joint_3"/>
      <geom type="sphere" size="0.12" rgba="1 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
 
class TowerDefenseVisualizer(Node):
 
    def __init__(self, model, data):
        super().__init__('tower_defense_visualizer')
 
        self.model = model
        self.data = data
 
        self.player_pos = np.array([3.0, 3.0])
        self.monster_positions = []
        self.bullet_positions = []
 
        self.create_subscription(
            String,
            '/tower_defense/debug',
            self._debug_callback,
            10
        )
 
        # Spieler Joint ID
        self.player_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, 'player_joint'
        )
 
        # Monster Joint IDs
        self.monster_joint_ids = []
        for i in range(MAX_MONSTERS):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f'monster_joint_{i}'
            )
            self.monster_joint_ids.append(jid)
 
        # Bullet Joint IDs
        self.bullet_joint_ids = []
        for i in range(4):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f'bullet_joint_{i}'
            )
            self.bullet_joint_ids.append(jid)
 
        self.get_logger().info('MuJoCo Visualizer gestartet!')
 
    def _debug_callback(self, msg):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
 
        # Spielerposition
        if 'player_x' in state and 'player_y' in state:
            self.player_pos = np.array([state['player_x'], state['player_y']])
 
        # Monsterposition
        self.monster_positions = []
        i = 0
        while f'm{i}_x' in state and f'm{i}_y' in state:
            self.monster_positions.append(
                np.array([state[f'm{i}_x'], state[f'm{i}_y']])
            )
            i += 1
 
        # Bullet Positionen
        self.bullet_positions = []
        i = 0
        while f'b{i}_x' in state and f'b{i}_y' in state:
            self.bullet_positions.append(
                np.array([state[f'b{i}_x'], state[f'b{i}_y']])
            )
            i += 1
 
    def update_mujoco(self):
        """Positionen in qpos schreiben."""
 
        # Spieler
        if self.player_joint_id >= 0:
            addr = self.model.jnt_qposadr[self.player_joint_id]
            self.data.qpos[addr + 0] = self.player_pos[0]
            self.data.qpos[addr + 1] = self.player_pos[1]
            self.data.qpos[addr + 2] = 0.1
            self.data.qpos[addr + 3] = 1.0
            self.data.qpos[addr + 4] = 0.0
            self.data.qpos[addr + 5] = 0.0
            self.data.qpos[addr + 6] = 0.0
 
        # Monster
        for i, jid in enumerate(self.monster_joint_ids):
            if jid >= 0:
                addr = self.model.jnt_qposadr[jid]
                if i < len(self.monster_positions):
                    self.data.qpos[addr + 0] = self.monster_positions[i][0]
                    self.data.qpos[addr + 1] = self.monster_positions[i][1]
                    self.data.qpos[addr + 2] = 0.1
                else:
                    self.data.qpos[addr + 2] = -1000.0
                self.data.qpos[addr + 3] = 1.0
                self.data.qpos[addr + 4] = 0.0
                self.data.qpos[addr + 5] = 0.0
                self.data.qpos[addr + 6] = 0.0
 
        # Bullets
        for i, jid in enumerate(self.bullet_joint_ids):
            if jid >= 0:
                addr = self.model.jnt_qposadr[jid]
                if i < len(self.bullet_positions):
                    self.data.qpos[addr + 0] = self.bullet_positions[i][0]
                    self.data.qpos[addr + 1] = self.bullet_positions[i][1]
                    self.data.qpos[addr + 2] = 0.1
                else:
                    self.data.qpos[addr + 2] = -1000.0
                self.data.qpos[addr + 3] = 1.0
                self.data.qpos[addr + 4] = 0.0
                self.data.qpos[addr + 5] = 0.0
                self.data.qpos[addr + 6] = 0.0
 
 
def main():
    rclpy.init()
 
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
 
    visualizer = TowerDefenseVisualizer(model, data)
 
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(visualizer),
        daemon=True
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
