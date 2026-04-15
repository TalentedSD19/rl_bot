"""
environment.py  --  HuskyTask2Env
Camera-only pick-and-place simulation using PyBullet.

Observation (10 floats):
  green_cx  green_cy  green_area  green_vis
  red_cx    red_cy    red_area    red_vis
  lift_norm  phase_norm

Actions (5 discrete)  -- movement only; lift/magnet are automated
  0 spin-L   1 spin-R   2 fwd   3 fwd+steer-L   4 fwd+steer-R
"""

import numpy as np
import pybullet as p
import pybullet_data

from config import (
    FORKLIFT_URDF,
    SPEED, LEFT_WHEELS, RIGHT_WHEELS,
    CAM_W, CAM_H, FOV,
    LIFT_JOINT, MAGNET_LINK, LIFT_STEP, LIFT_MIN, LIFT_MAX,
    BIG_R, BIG_H, SML_R, SML_H,
    PHASE_FIND_GREEN, PHASE_APPROACH_GREEN, PHASE_PICKUP,
    PHASE_FIND_RED, PHASE_APPROACH_RED, PHASE_DROP, N_PHASES,
    GREEN_CLOSE_AREA, RED_CLOSE_AREA, MAGNET_DIST, AUTO_REVERSE_STEPS,
)


class HuskyTask2Env:
    """Multi-phase pick-and-place. Front camera is the only sensor."""

    # Wheel velocity pairs keyed by discrete action
    VEL_MAP = {
        0: (-SPEED,       +SPEED),
        1: (+SPEED,       -SPEED),
        2: (+SPEED,       +SPEED),
        3: (+SPEED * 0.4, +SPEED),
        4: (+SPEED,       +SPEED * 0.4),
    }

    def __init__(self, gui: bool = False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        self.constraint_id = None
        self._build_world()

    # -- World setup ---------------------------------------------------------

    def _build_world(self):
        pid = self.client
        p.setGravity(0, 0, -9.81, physicsClientId=pid)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=pid)
        self.husky    = p.loadURDF("husky/husky.urdf", [0, 0, 0.15],
                                   physicsClientId=pid)
        self.mast     = p.loadURDF(FORKLIFT_URDF, basePosition=[0.6, 0, 0.8],
                                   useFixedBase=False, physicsClientId=pid)

        weld = p.createConstraint(self.husky, -1, self.mast, -1,
                                  p.JOINT_FIXED, [0, 0, 0],
                                  [0.6, 0, -0.15], [0, 0, -0.8],
                                  physicsClientId=pid)
        p.changeConstraint(weld, maxForce=50000, physicsClientId=pid)

        def make_cyl(mass, r, h, color):
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=pid)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                      rgbaColor=color, physicsClientId=pid)
            return p.createMultiBody(mass, col, vis, physicsClientId=pid)

        def make_hollow_box(mass, r, h, color, t=0.05):
            """Hollow open-top box: bottom plate + 4 walls, no lid."""
            orn = [0, 0, 0, 1]

            def box_col(hx, hy, hz):
                return p.createCollisionShape(p.GEOM_BOX,
                                              halfExtents=[hx, hy, hz],
                                              physicsClientId=pid)

            def box_vis(hx, hy, hz):
                return p.createVisualShape(p.GEOM_BOX,
                                           halfExtents=[hx, hy, hz],
                                           rgbaColor=color,
                                           physicsClientId=pid)

            # (halfExtents, position relative to body origin = box centre)
            parts = [
                ([r,     r,   t / 2],  [0,          0,          -h / 2 + t / 2]),  # bottom
                ([t / 2, r,   h / 2],  [ r - t / 2, 0,           0            ]),  # wall +X
                ([t / 2, r,   h / 2],  [-(r - t/2), 0,           0            ]),  # wall -X
                ([r - t, t/2, h / 2],  [0,           r - t / 2,  0            ]),  # wall +Y
                ([r - t, t/2, h / 2],  [0,          -(r - t/2),  0            ]),  # wall -Y
            ]

            cols = [box_col(*he) for he, _ in parts]
            viss = [box_vis(*he) for he, _ in parts]
            lpos = [pos          for _,  pos in parts]
            n    = len(parts)

            return p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=-1,
                basePosition=[0, 0, 0],
                baseOrientation=orn,
                linkMasses=[0.0] * n,
                linkCollisionShapeIndices=cols,
                linkVisualShapeIndices=viss,
                linkPositions=lpos,
                linkOrientations=[orn] * n,
                linkInertialFramePositions=[[0, 0, 0]] * n,
                linkInertialFrameOrientations=[orn] * n,
                linkParentIndices=[0] * n,
                linkJointTypes=[p.JOINT_FIXED] * n,
                linkJointAxis=[[0, 0, 1]] * n,
                physicsClientId=pid,
            )

        self.big_cyl   = make_hollow_box(0, BIG_R * 0.9, BIG_H, [1.0, 0.0, 0.0, 1.0])
        self.small_cyl = make_cyl(0.2, SML_R, SML_H, [0.0, 1.0, 0.0, 1.0])

    # -- Reset ---------------------------------------------------------------

    def reset(self) -> np.ndarray:
        pid = self.client

        p.resetBasePositionAndOrientation(self.husky, [0, 0, 0.15], [0, 0, 0, 1],
                                          physicsClientId=pid)
        p.resetBaseVelocity(self.husky, [0, 0, 0], [0, 0, 0], physicsClientId=pid)
        for j in range(p.getNumJoints(self.husky, physicsClientId=pid)):
            p.resetJointState(self.husky, j, 0, 0, physicsClientId=pid)

        p.resetBasePositionAndOrientation(self.mast, [0.6, 0, 0.8], [0, 0, 0, 1],
                                          physicsClientId=pid)
        p.resetBaseVelocity(self.mast, [0, 0, 0], [0, 0, 0], physicsClientId=pid)
        p.resetJointState(self.mast, LIFT_JOINT, 0, 0, physicsClientId=pid)

        if self.constraint_id is not None:
            try:
                p.removeConstraint(self.constraint_id, physicsClientId=pid)
            except Exception:
                pass
            self.constraint_id = None
        p.setCollisionFilterPair(self.small_cyl, self.plane_id, -1, -1,
                                 enableCollision=1, physicsClientId=pid)
        p.changeVisualShape(self.mast, MAGNET_LINK,
                            rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=pid)

        p.resetBasePositionAndOrientation(
            self.big_cyl, self._rand_pos(3.0, 6.0, BIG_H / 2), [0, 0, 0, 1],
            physicsClientId=pid)
        p.resetBasePositionAndOrientation(
            self.small_cyl, self._rand_pos(2.0, 4.0, SML_H / 2), [0, 0, 0, 1],
            physicsClientId=pid)

        self.phase         = PHASE_FIND_GREEN
        self.lift_target   = 0.0
        self.step_count    = 0
        self.auto_sub      = 0
        self.auto_counter  = 0
        self.magnet_on     = False
        self.task_success  = False          # True only if cylinder lands inside box
        self._reset_nav_trackers()

        for _ in range(20):
            p.stepSimulation(physicsClientId=pid)

        return self._get_obs()

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _rand_pos(lo: float, hi: float, z: float):
        a = np.random.uniform(0, 2 * np.pi)
        d = np.random.uniform(lo, hi)
        return [d * np.cos(a), d * np.sin(a), z]

    def _reset_nav_trackers(self):
        self.prev_visible       = False
        self.prev_area          = 0.0
        self.prev_centre_error  = 0.5
        self.discovery_rewarded = False

    def _update_nav_trackers(self, obs: np.ndarray):
        cx, _, area, visible = self._active_obs(obs)
        self.prev_visible      = bool(visible)
        if visible:
            self.prev_area         = area
            self.prev_centre_error = abs(cx - 0.5)

    # -- Actuation -----------------------------------------------------------

    def _drive(self, left: float, right: float):
        pid = self.client
        for j in LEFT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=left, force=100,
                                    physicsClientId=pid)
        for j in RIGHT_WHEELS:
            p.setJointMotorControl2(self.husky, j, p.VELOCITY_CONTROL,
                                    targetVelocity=right, force=100,
                                    physicsClientId=pid)

    def _update_lift(self):
        p.setJointMotorControl2(self.mast, LIFT_JOINT, p.POSITION_CONTROL,
                                targetPosition=self.lift_target, force=2000,
                                physicsClientId=self.client)

    # -- Magnet --------------------------------------------------------------

    def _magnet_pos(self) -> np.ndarray:
        return np.array(p.getLinkState(self.mast, MAGNET_LINK,
                                       physicsClientId=self.client)[0])

    def _try_attach(self) -> bool:
        if self.constraint_id is not None:
            return True
        cyl_pos = np.array(p.getBasePositionAndOrientation(
            self.small_cyl, physicsClientId=self.client)[0])
        if np.linalg.norm(self._magnet_pos() - cyl_pos) < MAGNET_DIST:
            self.constraint_id = p.createConstraint(
                self.mast, MAGNET_LINK, self.small_cyl, -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0],
                physicsClientId=self.client)
            p.changeConstraint(self.constraint_id, maxForce=500000,
                               physicsClientId=self.client)
            p.setCollisionFilterPair(self.small_cyl, self.plane_id, -1, -1,
                                     enableCollision=0, physicsClientId=self.client)
            return True
        return False

    def _detach(self):
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id, physicsClientId=self.client)
            self.constraint_id = None
            p.setCollisionFilterPair(self.small_cyl, self.plane_id, -1, -1,
                                     enableCollision=1, physicsClientId=self.client)

    # -- Camera --------------------------------------------------------------

    def _get_rgb(self) -> np.ndarray:
        pid = self.client
        pos, ori = p.getBasePositionAndOrientation(self.husky, physicsClientId=pid)
        rot    = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        eye    = np.array(pos) + rot @ [0.5, 0.0, 0.3]
        target = eye + rot @ [1.0, 0.0, 0.0]
        up     = rot @ [0.0, 0.0, 1.0]
        view   = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj   = p.computeProjectionMatrixFOV(FOV, CAM_W / CAM_H, 0.02, 20.0)
        _, _, rgb, _, _ = p.getCameraImage(CAM_W, CAM_H, view, proj,
                                           renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=pid)
        return np.array(rgb, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)[:, :, :3]

    @staticmethod
    def _detect(rgb: np.ndarray, green: bool = True):
        """Returns (cx, cy, area) normalised to frame, or None."""
        if green:
            mask = (rgb[:, :, 0] < 80) & (rgb[:, :, 1] > 150) & (rgb[:, :, 2] < 80)
        else:
            mask = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)
        n = int(mask.sum())
        if n < 5:
            return None
        ys, xs = np.where(mask)
        return xs.mean() / CAM_W, ys.mean() / CAM_H, n / (CAM_W * CAM_H)

    # -- Observation ---------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        rgb  = self._get_rgb()
        g    = self._detect(rgb, green=True)
        r    = self._detect(rgb, green=False)
        gobs = list(g) + [1.0] if g else [0.0, 0.0, 0.0, 0.0]
        robs = list(r) + [1.0] if r else [0.0, 0.0, 0.0, 0.0]
        lift = p.getJointState(self.mast, LIFT_JOINT,
                               physicsClientId=self.client)[0]
        lift_norm  = (lift - LIFT_MIN) / (LIFT_MAX - LIFT_MIN)
        phase_norm = self.phase / (N_PHASES - 1)
        return np.array(gobs + robs + [lift_norm, phase_norm], dtype=np.float32)

    def _active_obs(self, obs: np.ndarray):
        """(cx, cy, area, vis) for the current navigation target."""
        if self.phase in (PHASE_FIND_GREEN, PHASE_APPROACH_GREEN):
            return obs[0], obs[1], obs[2], obs[3]
        return obs[4], obs[5], obs[6], obs[7]

    # -- Step ----------------------------------------------------------------

    def step(self, action: int):
        """Returns (obs, reward, done)."""
        self.step_count += 1

        if self.phase in (PHASE_PICKUP, PHASE_DROP):
            return self._step_automated()

        self._drive(*self.VEL_MAP[action])
        for _ in range(4):
            self._update_lift()
            p.stepSimulation(physicsClientId=self.client)

        obs           = self._get_obs()
        reward, done  = self._reward(obs, action)
        self._update_nav_trackers(obs)
        return obs, reward, done

    # -- Automated phases ----------------------------------------------------

    def _step_automated(self):
        pid    = self.client
        reward = 0.0
        done   = False

        self._drive(0, 0)

        if self.phase == PHASE_PICKUP:
            if self.auto_sub == 0:
                self.lift_target = max(self.lift_target - LIFT_STEP, LIFT_MIN)
                if self.lift_target <= LIFT_MIN + 0.01:
                    self.auto_sub     = 1
                    self.auto_counter = 0

            elif self.auto_sub == 1:
                if not self.magnet_on:
                    self.magnet_on = True
                    p.changeVisualShape(self.mast, MAGNET_LINK,
                                        rgbaColor=[0.0, 1.0, 0.0, 1.0],
                                        physicsClientId=pid)
                if self._try_attach():
                    reward       += 5.0
                    self.auto_sub = 2
                else:
                    self.auto_counter += 1
                    if self.auto_counter > 15:
                        reward -= 3.0
                        self.magnet_on = False
                        p.changeVisualShape(self.mast, MAGNET_LINK,
                                            rgbaColor=[0.7, 0.7, 0.7, 1.0],
                                            physicsClientId=pid)
                        self.phase    = PHASE_APPROACH_GREEN
                        self.auto_sub = 0
                        self._reset_nav_trackers()

            elif self.auto_sub == 2:
                self.lift_target = min(self.lift_target + LIFT_STEP, LIFT_MAX)
                if self.lift_target >= LIFT_MAX - 0.01:
                    reward       += 10.0
                    self.phase    = PHASE_FIND_RED
                    self.auto_sub = 0
                    self._reset_nav_trackers()

        elif self.phase == PHASE_DROP:
            if self.auto_sub == 0:
                self._detach()
                self.magnet_on = False
                p.changeVisualShape(self.mast, MAGNET_LINK,
                                    rgbaColor=[0.7, 0.7, 0.7, 1.0],
                                    physicsClientId=pid)
                reward          += 5.0
                self.auto_sub    = 1
                self.auto_counter = 0

            elif self.auto_sub == 1:
                self._drive(-SPEED, -SPEED)
                self.auto_counter += 1
                if self.auto_counter >= AUTO_REVERSE_STEPS:
                    self.auto_sub = 2

            elif self.auto_sub == 2:
                self.lift_target = max(self.lift_target - LIFT_STEP, 0.0)
                if self.lift_target <= 0.01:
                    # Let physics settle for a moment before checking position
                    for _ in range(30):
                        p.stepSimulation(physicsClientId=pid)
                    if self._cylinder_inside_box():
                        reward             += 100.0
                        self.task_success   = True
                    else:
                        reward -= 30.0      # reached drop phase but missed the box
                    done = True

        for _ in range(4):
            self._update_lift()
            p.stepSimulation(physicsClientId=pid)

        return self._get_obs(), reward, done

    # -- Reward --------------------------------------------------------------

    def _cylinder_inside_box(self) -> bool:
        """
        Returns True only when the green cylinder has physically landed
        inside the red hollow box.

        Geometry (from config / make_hollow_box):
          box half-extent  r = BIG_R * 0.9  = 0.63 m
          wall thickness   t = 0.05 m
          inner clearance      = r - t       = 0.58 m
          cylinder radius      = SML_R       = 0.16 m
          max centre-offset    = 0.58 - 0.16 = 0.42 m   (horizontal)

          box top z  = box_z + BIG_H / 2               (≈ 0.50 m from ground)
          cylinder must be below box top to be "inside", not sitting on walls.
        """
        cyl_pos = np.array(
            p.getBasePositionAndOrientation(
                self.small_cyl, physicsClientId=self.client)[0])
        box_pos = np.array(
            p.getBasePositionAndOrientation(
                self.big_cyl, physicsClientId=self.client)[0])

        horiz_dist = np.linalg.norm(cyl_pos[:2] - box_pos[:2])
        box_top_z  = box_pos[2] + BIG_H / 2

        inner_clearance = BIG_R * 0.9 - 0.05           # 0.58 m
        max_offset      = inner_clearance - SML_R       # 0.42 m

        return horiz_dist <= max_offset and cyl_pos[2] < box_top_z

    def _reward(self, obs: np.ndarray, action: int):
        cx, _, area, visible = self._active_obs(obs)
        visible = bool(visible)
        r       = 0.0

        if not visible:
            if self.prev_visible:
                r -= 5.0
            if action == 2:
                r -= 1.0
            elif action in (3, 4):
                r -= 0.5
            r -= 0.05
            return r, False

        if not self.prev_visible and not self.discovery_rewarded:
            r += 3.0
            self.discovery_rewarded = True

        centre_error = abs(cx - 0.5)

        r += (self.prev_centre_error - centre_error) * 5.0
        r -= centre_error * 0.5

        if cx > 0.5 and action in (1, 4):
            r += centre_error * 2.0
        elif cx < 0.5 and action in (0, 3):
            r += centre_error * 2.0

        if self.phase in (PHASE_FIND_GREEN, PHASE_FIND_RED):
            if centre_error < 0.30:
                self.phase = (PHASE_APPROACH_GREEN if self.phase == PHASE_FIND_GREEN
                              else PHASE_APPROACH_RED)
                r += 1.0

        elif self.phase in (PHASE_APPROACH_GREEN, PHASE_APPROACH_RED):
            close_area = (GREEN_CLOSE_AREA if self.phase == PHASE_APPROACH_GREEN
                          else RED_CLOSE_AREA)

            if centre_error < 0.12:
                delta_area = float(np.clip(area - self.prev_area, -0.05, 0.05))
                r += delta_area * 30.0
            elif centre_error < 0.20:
                delta_area = float(np.clip(area - self.prev_area, -0.05, 0.05))
                r += delta_area * 10.0
            else:
                if action in (2, 3, 4):
                    r -= 0.4

            if self.phase == PHASE_APPROACH_GREEN and area >= GREEN_CLOSE_AREA * 0.4:
                r += max(0.0, 0.15 - centre_error) * 6.0
            elif self.phase == PHASE_APPROACH_RED and area >= RED_CLOSE_AREA * 0.4:
                r += max(0.0, 0.15 - centre_error) * 6.0

            if area >= close_area and centre_error < 0.10:
                r += 10.0
                self.phase     = (PHASE_PICKUP if self.phase == PHASE_APPROACH_GREEN
                                  else PHASE_DROP)
                self.auto_sub     = 0
                self.auto_counter = 0

        r -= 0.05
        return r, False

    # -- Cleanup -------------------------------------------------------------

    def close(self):
        p.disconnect(self.client)
