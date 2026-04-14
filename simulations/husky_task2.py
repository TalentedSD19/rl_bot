import os
import pybullet as p
import pybullet_data
import time
import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
FORKLIFT_URDF = os.path.join(DIR, "forklift_mast.urdf")

# ── Setup ──────────────────────────────────────────────────────────────────────
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
husky    = p.loadURDF("husky/husky.urdf", [0, 0, 0.15])
mast     = p.loadURDF(FORKLIFT_URDF, basePosition=[0.6, 0, 0.8], useFixedBase=False)

# Joint / link indices on the mast
LIFT_JOINT  = 0
MAGNET_LINK = 2

# Weld mast rigidly to the Husky's front
weld = p.createConstraint(husky, -1, mast, -1, p.JOINT_FIXED, [0,0,0],
                           [0.6, 0, -0.15], [0, 0, -0.8])
p.changeConstraint(weld, maxForce=50000)

# ── Scene objects ──────────────────────────────────────────────────────────────
rng = np.random.default_rng()

def spawn_cylinder(mass, radius, height, color):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(mass, col, vis)

def rand_pos(dist_range, z):
    a, d = rng.uniform(0, 2*np.pi), rng.uniform(*dist_range)
    return [d*np.cos(a), d*np.sin(a), z]

BIG_R, BIG_H   = 0.4, 0.5
SML_R, SML_H   = BIG_R/2.5, BIG_H/2.5

big_cyl   = spawn_cylinder(0,   BIG_R, BIG_H, [1.0, 0.0, 0.0, 1.0])
small_cyl = spawn_cylinder(0.2, SML_R, SML_H, [0.0, 1.0, 0.0, 1.0])

p.resetBasePositionAndOrientation(big_cyl,   rand_pos((3.0, 5.0), BIG_H/2), [0,0,0,1])
p.resetBasePositionAndOrientation(small_cyl, rand_pos((2.0, 4.0), SML_H/2), [0,0,0,1])

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H  = 320, 240
CAM_OFFSET    = np.array([0.5, 0, 0.3])   # metres ahead & above Husky base
CAM_PROJ      = p.computeProjectionMatrixFOV(fov=90, aspect=CAM_W/CAM_H,
                                              nearVal=0.1, farVal=20.0)

def get_camera_image():
    pos, ori = p.getBasePositionAndOrientation(husky)
    rot    = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    eye    = np.array(pos) + rot @ CAM_OFFSET
    target = eye + rot @ [1, 0, 0]
    up     = rot @ [0, 0, 1]
    view   = p.computeViewMatrix(eye, target, up)
    _, _, rgb, depth, _ = p.getCameraImage(CAM_W, CAM_H, view, CAM_PROJ,
                                           renderer=p.ER_TINY_RENDERER)
    return np.array(rgb, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3], depth

# ── Magnet ────────────────────────────────────────────────────────────────────
magnet_on     = False
constraint_id = None
p.changeVisualShape(mast, MAGNET_LINK, rgbaColor=[0.7, 0.7, 0.7, 1.0])

def magnet_pos():
    return np.array(p.getLinkState(mast, MAGNET_LINK)[0])

def try_attach():
    global constraint_id
    if constraint_id is not None:
        return
    if np.linalg.norm(magnet_pos() - np.array(p.getBasePositionAndOrientation(small_cyl)[0])) < 0.5:
        constraint_id = p.createConstraint(
            mast, MAGNET_LINK, small_cyl, -1,
            p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
        p.changeConstraint(constraint_id, maxForce=500000)
        p.setCollisionFilterPair(small_cyl, plane_id, -1, -1, enableCollision=0)

def detach():
    global constraint_id
    if constraint_id is not None:
        p.removeConstraint(constraint_id)
        constraint_id = None
        p.setCollisionFilterPair(small_cyl, plane_id, -1, -1, enableCollision=1)

# ── Drive ─────────────────────────────────────────────────────────────────────
LEFT_WHEELS  = [2, 4]
RIGHT_WHEELS = [3, 5]
SPEED        = 7

def drive(left, right):
    for j in LEFT_WHEELS:
        p.setJointMotorControl2(husky, j, p.VELOCITY_CONTROL, targetVelocity=left,  force=100)
    for j in RIGHT_WHEELS:
        p.setJointMotorControl2(husky, j, p.VELOCITY_CONTROL, targetVelocity=right, force=100)

# ── Key bindings ──────────────────────────────────────────────────────────────
DRIVE_KEYS = {
    p.B3G_UP_ARROW:    ( SPEED,  SPEED),
    p.B3G_DOWN_ARROW:  (-SPEED, -SPEED),
    p.B3G_LEFT_ARROW:  (-SPEED,  SPEED),
    p.B3G_RIGHT_ARROW: ( SPEED, -SPEED),
}
M_KEY      = ord('m')
LIFT_UP    = ord('p')
LIFT_DOWN  = ord('l')
LIFT_SPEED = 0.3
lift_target = 0.0

# ── Main loop ─────────────────────────────────────────────────────────────────
step = 0
while True:
    keys = p.getKeyboardEvents()

    # Drive
    l = r = 0
    for k, (lv, rv) in DRIVE_KEYS.items():
        if k in keys and keys[k] & p.KEY_IS_DOWN:
            l, r = lv, rv
            break
    drive(l, r)

    # Lift
    if LIFT_UP   in keys and keys[LIFT_UP]   & p.KEY_IS_DOWN:
        lift_target = min(lift_target + LIFT_SPEED / 240, 0.4)
    if LIFT_DOWN in keys and keys[LIFT_DOWN] & p.KEY_IS_DOWN:
        lift_target = max(lift_target - LIFT_SPEED / 240, -0.4)
    p.setJointMotorControl2(mast, LIFT_JOINT, p.POSITION_CONTROL,
                            targetPosition=lift_target, force=2000)

    # Magnet toggle
    if M_KEY in keys and keys[M_KEY] & p.KEY_WAS_TRIGGERED:
        magnet_on = not magnet_on
        p.changeVisualShape(mast, MAGNET_LINK,
                            rgbaColor=[0.0,1.0,0.0,1.0] if magnet_on else [0.7, 0.7, 0.7, 1.0])
        if not magnet_on:
            detach()
    if magnet_on:
        try_attach()

    # Camera (every 10 steps)
    if step % 10 == 0:
        rgb, depth = get_camera_image()
        if HAVE_CV2:
            cv2.imshow("Husky Front Camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    p.stepSimulation()
    time.sleep(1 / 240)
    step += 1
