# run_demo.py

import subprocess
import sys
import time

processes = []

try:
    publisher = subprocess.Popen([sys.executable, "test_ros2.py"])
    processes.append(publisher)

    time.sleep(2)

    visualizer = subprocess.Popen([sys.executable, "mujoco_viz.py"])
    processes.append(visualizer)

    visualizer.wait()

except KeyboardInterrupt:
    print("Beende Demo...")

finally:
    for p in processes:
        if p.poll() is None:
            p.terminate()