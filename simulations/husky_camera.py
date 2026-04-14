import pybullet as p
import pybullet_data
import time
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────────
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

LEFT_WHEELS  = [2, 4]
RIGHT_WHEELS = [3, 5]
SPEED = 7

# ── Red cylinder (static — mass=0 so it won't move when touched) ───────────────
CYL_RADIUS = 0.4
CYL_HEIGHT = 1.0

col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=CYL_RADIUS, height=CYL_HEIGHT)
vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=CYL_RADIUS, length=CYL_HEIGHT,
                              rgbaColor=[1.0, 0.0, 0.0, 1.0])  # red
box_id = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=col_id,
                            baseVisualShapeIndex=vis_id,
                            basePosition=[0, 0, 0])

# Spawn at a random angle, fixed distance from robot
rng      = np.random.default_rng()
angle    = rng.uniform(0, 2 * np.pi)
dist     = rng.uniform(3.0, 5.0)
box_pos  = [dist * np.cos(angle), dist * np.sin(angle), CYL_HEIGHT / 2]
p.resetBasePositionAndOrientation(box_id, box_pos, [0, 0, 0, 1])

# ── Camera ────────────────────────────────────────────────────────────────────
CAM        = dict(width=320, height=240, fov=90, near=0.1, far=20.0)
CAM_OFFSET = np.array([0.5, 0, 0.3])

def get_camera_image():
    pos, ori = p.getBasePositionAndOrientation(husky)
    rot      = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    eye      = np.array(pos) + rot @ CAM_OFFSET
    target   = eye + rot @ [1, 0, 0]
    up       = rot @ [0, 0, 1]
    view     = p.computeViewMatrix(eye, target, up)
    proj     = p.computeProjectionMatrixFOV(CAM["fov"], CAM["width"] / CAM["height"],
                                            CAM["near"], CAM["far"])
    _, _, rgb, depth, _ = p.getCameraImage(CAM["width"], CAM["height"], view, proj,
                                           renderer=p.ER_TINY_RENDERER)
    return np.array(rgb, dtype=np.uint8).reshape(CAM["height"], CAM["width"], 4)[:, :, :3], depth

def drive(left_vel, right_vel):
    for j in LEFT_WHEELS:
        p.setJointMotorControl2(husky, j, p.VELOCITY_CONTROL, targetVelocity=left_vel,  force=100)
    for j in RIGHT_WHEELS:
        p.setJointMotorControl2(husky, j, p.VELOCITY_CONTROL, targetVelocity=right_vel, force=100)

# ── Main loop ─────────────────────────────────────────────────────────────────
KEY_TO_DRIVE = {
    p.B3G_UP_ARROW:    ( SPEED,  SPEED),
    p.B3G_DOWN_ARROW:  (-SPEED, -SPEED),
    p.B3G_LEFT_ARROW:  (-SPEED,  SPEED),
    p.B3G_RIGHT_ARROW: ( SPEED, -SPEED),
}

step = 0
while True:
    keys = p.getKeyboardEvents()
    left_vel = right_vel = 0
    for key, (l, r) in KEY_TO_DRIVE.items():
        if key in keys:
            left_vel, right_vel = l, r
            break

    drive(left_vel, right_vel)
    p.stepSimulation()

    if step % 10 == 0:
        rgb, depth = get_camera_image()

    step += 1
    time.sleep(1 / 240)