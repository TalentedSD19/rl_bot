import pybullet as p
import pybullet_data
import time

# Connect
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load environment
planeId = p.loadURDF("plane.urdf")

huskyStartPos = [0, 0, 0.1]
huskyStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
husky = p.loadURDF("husky/husky.urdf", huskyStartPos, huskyStartOrientation)

# Husky wheel joints
wheels = [2, 3, 4, 5]

# Speed settings
forward_speed = 4
turn_speed = 4

while True:
    keys = p.getKeyboardEvents()

    leftVel = 0
    rightVel = 0

    if p.B3G_UP_ARROW in keys:
        leftVel = forward_speed
        rightVel = forward_speed

    if p.B3G_DOWN_ARROW in keys:
        leftVel = -forward_speed
        rightVel = -forward_speed

    if p.B3G_LEFT_ARROW in keys:
        leftVel = -turn_speed
        rightVel = turn_speed

    if p.B3G_RIGHT_ARROW in keys:
        leftVel = turn_speed
        rightVel = -turn_speed

    # Apply velocity to wheels
    for i in [2, 4]:  # left wheels
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=leftVel, force=100)

    for i in [3, 5]:  # right wheels
        p.setJointMotorControl2(husky, i, p.VELOCITY_CONTROL, targetVelocity=rightVel, force=100)

    p.stepSimulation()
    time.sleep(1./240.)