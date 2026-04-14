import pybullet as p
import pybullet_data
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load ground plane
planeId = p.loadURDF("plane.urdf")

# Load Husky
huskyStartPos = [0, 0, 0.1]
huskyStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
huskyId = p.loadURDF("husky/husky.urdf", huskyStartPos, huskyStartOrientation)

# Simulation loop
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
