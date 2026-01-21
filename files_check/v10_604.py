# tidybot_nav_env_v10.py
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TidybotNavEnvV10(gym.Env):
    """
    V10 (Thesis-level): Base + Arm staged objective with Multi-Modal Observations.

    Stage 0: Base navigates to base_goal_site (arm held).
    Stage 1: Arm moves EE (pinch_site) to arm_goal_site (base held).
    Done: both satisfied.

    Observation (Dict):
      - "image": RGB from wrist camera (uint8, HWC)
      - "state": vector features (float32)

    Action (10D):
      [vx_body, vy_body, wz, dq1..dq7]
      - base cmd in body frame
      - arm cmd as joint delta-rate (rad/s-ish), integrated into actuator targets

    Notes:
      - Works with SB3 MultiInputPolicy (CNN on image + MLP on state).
      - Includes: time penalty, safety layer, domain randomization, configurable reward weights, rich logging.
    """

    metadata = {"render_modes": ["human", None]}

    # --- XML names ---
    BASE_GOAL_SITE = "base_goal_site"
    ARM_GOAL_SITE = "arm_goal_site"
    PINCH_SITE = "pinch_site"

    WRIST_CAM = "wrist"   # important
    # BASE_CAM = "base"   # optional (not used here, but you can extend)

    def __init__(
        self,
        xml_path="tidybot.xml",
        render_mode=None,
        # simulation / timing
        dt: float = 0.05,
        max_steps: int = 600,
        xy_limit: float = 5.0,
        # base limits
        max_vx: float = 0.12,
        max_vy: float = 0.08,
        max_wz: float = 0.25,
        # arm limits
        max_dq=None,  # array-like length 7
        # goal thresholds
        base_reach_radius: float = 0.35,
        base_heading_tol: float = 0.8,
        arm_reach_radius: float = 0.08,
        # camera
        cam_w: int = 160,
        cam_h: int = 120,
        # lidar
        lidar_max_range: float = 8.0,
        # domain randomization
        spawn_xy_range: float = 2.0,
        spawn_yaw_range: float = math.pi,
        # safety
        safety_lidar_stop_m: float = 0.30,
        safety_lidar_scale: float = 0.50,
        safety_joint_margin: float = 0.08,   # rad margin to joint limit to start scaling
        safety_joint_scale: float = 0.30,
        # reward shaping weights
        reward_weights=None,
        # time penalty
        step_time_penalty: float = -0.01,
        # smoothing
        tau: float = 0.25,
    ):
        super().__init__()
        self.xml_path = xml_path
        self.render_mode = render_mode

        # ---------- MuJoCo ----------
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # ---------- Config ----------
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.xy_limit = float(xy_limit)
        self.spawn_xy_range = float(spawn_xy_range)
        self.spawn_yaw_range = float(spawn_yaw_range)

        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)

        if max_dq is None:
            self.max_dq = np.array([0.25, 0.25, 0.25, 0.35, 0.45, 0.45, 0.45], dtype=np.float32)
        else:
            arr = np.asarray(max_dq, dtype=np.float32)
            assert arr.shape == (7,)
            self.max_dq = arr

        self.base_reach_radius = float(base_reach_radius)
        self.base_heading_tol = float(base_heading_tol)
        self.arm_reach_radius = float(arm_reach_radius)

        self.image_w = int(image_w)
        self.image_h = int(image_h)

        self.lidar_max_range = float(lidar_max_range)

        self.safety_lidar_stop_m = float(safety_lidar_stop_m)
        self.safety_lidar_scale = float(safety_lidar_scale)
        self.safety_joint_margin = float(safety_joint_margin)
        self.safety_joint_scale = float(safety_joint_scale)

        self.step_time_penalty = float(step_time_penalty)
        self.tau = float(tau)

        # ---------- Reward weights ----------
        # You can tune these from train config without editing the env
        default_rw = dict(
            # stage 0 (base)
            base_prog=2.0,
            base_dist=0.20,
            base_heading=0.05,
            base_ctrl=0.05,
            base_accel=0.02,

            # stage 1 (arm)
            ee_prog=3.0,
            ee_dist=0.40,
            arm_ctrl=0.02,
            arm_accel=0.01,

            # shared
            collision=-8.0,
            success=+40.0,
            timeout=-6.0,
        )
        if reward_weights is None:
            self.rw = default_rw
        else:
            self.rw = default_rw.copy()
            self.rw.update(dict(reward_weights))

        # ---------- Base joints ----------
        jid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
        jid_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y")
        jid_th = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_th")
        if min(jid_x, jid_y, jid_th) < 0:
            raise RuntimeError("Missing base joints: joint_x / joint_y / joint_th")

        self.qx = self.model.jnt_qposadr[jid_x]
        self.qy = self.model.jnt_qposadr[jid_y]
        self.qyaw = self.model.jnt_qposadr[jid_th]

        # Base actuators (position actuators)
        self.aid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_x")
        self.aid_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_y")
        self.aid_th = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_th")
        if min(self.aid_x, self.aid_y, self.aid_th) < 0:
            raise RuntimeError("Missing base actuators: joint_x / joint_y / joint_th (actuators)")

        # ---------- Arm joints + actuators ----------
        self.arm_joint_names = [f"joint_{i}" for i in range(1, 8)]
        self.arm_act_ids = []
        self.arm_jposadr = []
        self.arm_jrange = []
        for jn in self.arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"Missing arm joint/actuator named '{jn}'")
            self.arm_act_ids.append(aid)
            self.arm_jposadr.append(self.model.jnt_qposadr[jid])
            self.arm_jrange.append(self.model.jnt_range[jid].copy())

        self.arm_act_ids = np.array(self.arm_act_ids, dtype=np.int32)
        self.arm_jposadr = np.array(self.arm_jposadr, dtype=np.int32)
        self.arm_jrange = np.array(self.arm_jrange, dtype=np.float32)

        # ---------- Sites ----------
        self.base_goal_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.BASE_GOAL_SITE)
        self.arm_goal_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ARM_GOAL_SITE)
        self.ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.PINCH_SITE)
        if self.base_goal_sid < 0:
            raise RuntimeError(f"Missing site '{self.BASE_GOAL_SITE}'")
        if self.arm_goal_sid < 0:
            raise RuntimeError(f"Missing site '{self.ARM_GOAL_SITE}'")
        if self.ee_sid < 0:
            raise RuntimeError(f"Missing site '{self.PINCH_SITE}'")

        # ---------- Camera ----------
        self.wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.WRIST_CAM)
        if self.wrist_cam_id < 0:
            raise RuntimeError(f"Missing camera '{self.WRIST_CAM}' in XML")

        # Offscreen renderer
        self._renderer = mujoco.Renderer(self.model, height=self.image_h, width=self.image_w)

        # ---------- Lidar sensors ----------
        # Keep same names as your v1-9 list; edit once if your XML differs.
        self.lidar_sensor_names = [
            "rfA_000", "rfA_045", "rfA_090", "rfA_135", "rfA_180",
            "rfB_000", "rfB_090", "rfB_180",
        ]
        self.lidar_indices = []
        for name in self.lidar_sensor_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise RuntimeError(f"Lidar sensor '{name}' not found in model")
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            if dim != 1:
                raise RuntimeError(f"Lidar sensor '{name}' has dim {dim}, expected 1")
            self.lidar_indices.append(adr)
        self.lidar_indices = np.array(self.lidar_indices, dtype=np.int32)

        # ---------- Internal state ----------
        self.viewer = None
        self.step_count = 0
        self.stage = 0  # 0 base, 1 arm

        self.vcur_base = np.zeros(3, dtype=np.float32)
        self.prev_base = np.zeros(3, dtype=np.float32)

        self.vcur_arm = np.zeros(7, dtype=np.float32)
        self.prev_arm = np.zeros(7, dtype=np.float32)

        self.arm_target = np.zeros(7, dtype=np.float32)

        self.last_base_dist = None
        self.last_ee_dist = None

        self.safety_interventions = 0

        # ---------- Spaces ----------
        # Action: 10D
        low = np.concatenate(
            [np.array([-self.max_vx, -self.max_vy, -self.max_wz], dtype=np.float32), -self.max_dq]
        )
        high = np.concatenate(
            [np.array([+self.max_vx, +self.max_vy, +self.max_wz], dtype=np.float32), +self.max_dq]
        )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Obs: Dict(image, state)
        # image: uint8 HWC
        img_space = spaces.Box(low=0, high=255, shape=(self.image_h, self.image_w, 3), dtype=np.uint8)

        # state vector: base errors + ee errors + stage + base vel + lidar + joint limit margins
        # base: x,y,yaw, dx,dy, dist, heading_err  -> 7
        # ee: ex,ey,ez, edx,edy,edz, dist_ee      -> 7
        # stage                               -> 1
        # base vel                            -> 3
        # lidar normalized                    -> 8
        # joint limit proximity (7)           -> 7
        # total = 33
        self.state_dim = 33
        state_space = spaces.Box(low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict({"image": img_space, "state": state_space})

    # --------------------- Helpers ---------------------
    def _base_pose(self):
        x = float(self.data.qpos[self.qx])
        y = float(self.data.qpos[self.qy])
        yaw = float(self.data.qpos[self.qyaw])
        return x, y, yaw

    def _site_pos(self, sid: int):
        return self.data.site_xpos[sid].copy()

    def _any_collision(self) -> bool:
        for c in self.data.contact:
            if c.dist < 0:
                return True
        return False

    def _get_lidar_norm(self):
        vals = []
        for adr in self.lidar_indices:
            r = float(self.data.sensordata[adr])
            if r < 0:
                r = self.lidar_max_range
            r = min(r, self.lidar_max_range)
            vals.append(r / self.lidar_max_range)  # 1.0 free, 0.0 close
        return np.array(vals, dtype=np.float32)

    def _lidar_min_m(self):
        # convert normalized back to meters
        ln = self._get_lidar_norm()
        return float(np.min(ln) * self.lidar_max_range)

    def _joint_limit_proximity(self):
        # returns per-joint distance to nearest limit (smaller => closer to limit)
        q = self.data.qpos[self.arm_jposadr].astype(np.float32)
        lo = self.arm_jrange[:, 0]
        hi = self.arm_jrange[:, 1]
        dist_lo = np.abs(q - lo)
        dist_hi = np.abs(hi - q)
        prox = np.minimum(dist_lo, dist_hi)
        return prox.astype(np.float32)  # shape (7,)

    def _compute_base_errors(self):
        goal = self._site_pos(self.base_goal_sid)
        x, y, yaw = self._base_pose()
        dx = float(goal[0] - x)
        dy = float(goal[1] - y)
        dist = float(math.hypot(dx, dy))
        desired = math.atan2(dy, dx)
        heading_err = float(wrap_to_pi(desired - yaw))
        return dx, dy, dist, heading_err

    def _compute_ee_errors(self):
        g = self._site_pos(self.arm_goal_sid)
        ee = self._site_pos(self.ee_sid)
        dx = float(g[0] - ee[0])
        dy = float(g[1] - ee[1])
        dz = float(g[2] - ee[2])
        dist = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        return ee, (dx, dy, dz), dist

    def _render_wrist_rgb(self):
        self._renderer.update_scene(self.data, camera=self.wrist_cam_id)
        img = self._renderer.render()
        # renderer gives RGB uint8 already
        return img

    def _build_obs(self):
        x, y, yaw = self._base_pose()
        dx, dy, dist_b, heading_err = self._compute_base_errors()
        ee, (edx, edy, edz), dist_ee = self._compute_ee_errors()

        lidar_norm = self._get_lidar_norm()
        joint_prox = self._joint_limit_proximity()

        # Pack state vector
        state = np.concatenate([
            # base 7
            np.array([x, y, yaw, dx, dy, dist_b, heading_err], dtype=np.float32),
            # ee 7
            np.array([ee[0], ee[1], ee[2], edx, edy, edz, dist_ee], dtype=np.float32),
            # stage 1
            np.array([float(self.stage)], dtype=np.float32),
            # base vel 3
            self.vcur_base.astype(np.float32),
            # lidar 8
            lidar_norm.astype(np.float32),
            # joint prox 7 (meters/radians scale; keep raw small values)
            joint_prox.astype(np.float32),
        ], axis=0)

        if state.shape != (self.state_dim,):
            raise RuntimeError(f"State dim mismatch: got {state.shape}, expected {(self.state_dim,)}")

        obs = {
            "image": self._render_wrist_rgb(),
            "state": state,
        }
        return obs

    def _safety_check(self, cmd_base: np.ndarray, cmd_arm: np.ndarray):
        """
        Simple intervention:
          - if lidar too close => slow base
          - if joints near limits => slow arm
        """
        interventions = 0

        # lidar
        lidar_min = self._lidar_min_m()
        if lidar_min < self.safety_lidar_stop_m:
            cmd_base *= self.safety_lidar_scale
            interventions += 1

        # joints
        prox = self._joint_limit_proximity()
        if float(np.min(prox)) < self.safety_joint_margin:
            cmd_arm *= self.safety_joint_scale
            interventions += 1

        return cmd_base, cmd_arm, lidar_min, prox, interventions

    # --------------------- Gym API ---------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Domain randomization (base spawn)
        x0 = float(self.np_random.uniform(-self.spawn_xy_range, self.spawn_xy_range))
        y0 = float(self.np_random.uniform(-self.spawn_xy_range, self.spawn_xy_range))
        yaw0 = float(self.np_random.uniform(-self.spawn_yaw_range, self.spawn_yaw_range))

        self.data.qpos[self.qx] = x0
        self.data.qpos[self.qy] = y0
        self.data.qpos[self.qyaw] = yaw0

        # Hold base targets at current pose
        self.data.ctrl[self.aid_x] = float(x0)
        self.data.ctrl[self.aid_y] = float(y0)
        self.data.ctrl[self.aid_th] = float(yaw0)

        # Forward once so sites/kinematics are correct
        mujoco.mj_forward(self.model, self.data)

        # Arm target = current qpos
        cur = self.data.qpos[self.arm_jposadr].astype(np.float32)
        self.arm_target[:] = cur
        for i, aid in enumerate(self.arm_act_ids):
            self.data.ctrl[aid] = float(self.arm_target[i])

        # internal state
        self.step_count = 0
        self.stage = 0

        self.vcur_base[:] = 0.0
        self.prev_base[:] = 0.0
        self.vcur_arm[:] = 0.0
        self.prev_arm[:] = 0.0

        self.safety_interventions = 0

        dx, dy, dist_b, _ = self._compute_base_errors()
        _, _, dist_ee = self._compute_ee_errors()
        self.last_base_dist = float(dist_b)
        self.last_ee_dist = float(dist_ee)

        obs = self._build_obs()
        info = {
            "stage": int(self.stage),
            "dist_base": float(dist_b),
            "dist_ee": float(dist_ee),
            "spawn": (x0, y0, yaw0),
        }
        return obs, info

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        cmd_base = action[:3].astype(np.float32)
        cmd_arm = action[3:].astype(np.float32)

        # Stage gating
        if self.stage == 0:
            cmd_arm[:] = 0.0
        else:
            cmd_base[:] = 0.0

        # Smooth
        alpha = self.dt / (self.tau + self.dt)
        self.vcur_base = (1 - alpha) * self.vcur_base + alpha * cmd_base
        self.vcur_arm = (1 - alpha) * self.vcur_arm + alpha * cmd_arm

        accel_base = self.vcur_base - self.prev_base
        accel_arm = self.vcur_arm - self.prev_arm
        self.prev_base[:] = self.vcur_base
        self.prev_arm[:] = self.vcur_arm

        # Safety layer
        safe_base, safe_arm, lidar_min_m, joint_prox, nint = self._safety_check(
            self.vcur_base.copy(), self.vcur_arm.copy()
        )
        if nint > 0:
            self.safety_interventions += nint
        self.vcur_base[:] = safe_base
        self.vcur_arm[:] = safe_arm

        # ---------------- Apply BASE (kinematic integration + position target) ----------------
        vx_b, vy_b, wz = self.vcur_base
        x, y, yaw = self._base_pose()
        cy, sy = math.cos(yaw), math.sin(yaw)

        vx_w = cy * vx_b - sy * vy_b
        vy_w = sy * vx_b + cy * vy_b

        x = float(np.clip(x + vx_w * self.dt, -self.xy_limit, self.xy_limit))
        y = float(np.clip(y + vy_w * self.dt, -self.xy_limit, self.xy_limit))
        yaw = float(yaw + wz * self.dt)

        self.data.qpos[self.qx] = x
        self.data.qpos[self.qy] = y
        self.data.qpos[self.qyaw] = yaw

        # hold base with actuators at integrated pose
        self.data.ctrl[self.aid_x] = float(x)
        self.data.ctrl[self.aid_y] = float(y)
        self.data.ctrl[self.aid_th] = float(yaw)

        # ---------------- Apply ARM (position targets) ----------------
        self.arm_target[:] = self.arm_target + self.vcur_arm * self.dt
        self.arm_target[:] = np.clip(self.arm_target, self.arm_jrange[:, 0], self.arm_jrange[:, 1])
        for i, aid in enumerate(self.arm_act_ids):
            self.data.ctrl[aid] = float(self.arm_target[i])

        # step physics
        mujoco.mj_step(self.model, self.data)

        # ---------------- Metrics ----------------
        dx, dy, dist_b, heading_err = self._compute_base_errors()
        _, _, dist_ee = self._compute_ee_errors()

        prog_b = float(self.last_base_dist - dist_b)
        prog_ee = float(self.last_ee_dist - dist_ee)
        self.last_base_dist = float(dist_b)
        self.last_ee_dist = float(dist_ee)

        collided = self._any_collision()

        # Threshold consistency: use self.base_reach_radius, self.base_heading_tol, self.arm_reach_radius
        base_reached = (dist_b < self.base_reach_radius) and (abs(heading_err) < self.base_heading_tol)
        if self.stage == 0 and base_reached:
            self.stage = 1

        arm_reached = (dist_ee < self.arm_reach_radius)
        reached = bool(base_reached and arm_reached)

        # ---------------- Reward ----------------
        reward = 0.0

        # explicit time penalty
        reward += self.step_time_penalty

        if self.stage == 0:
            reward += self.rw["base_prog"] * prog_b
            reward += -self.rw["base_dist"] * dist_b
            reward += -self.rw["base_heading"] * abs(heading_err)
            reward += -self.rw["base_ctrl"] * float(np.dot(self.vcur_base, self.vcur_base))
            reward += -self.rw["base_accel"] * float(np.dot(accel_base, accel_base))
        else:
            reward += self.rw["ee_prog"] * prog_ee
            reward += -self.rw["ee_dist"] * dist_ee
            reward += -self.rw["arm_ctrl"] * float(np.dot(self.vcur_arm, self.vcur_arm))
            reward += -self.rw["arm_accel"] * float(np.dot(accel_arm, accel_arm))

        if collided:
            reward += self.rw["collision"]

        if reached:
            reward += self.rw["success"]

        timeout = self.step_count >= self.max_steps
        terminated = bool(reached)
        truncated = bool(timeout and not reached)
        if truncated:
            reward += self.rw["timeout"]

        obs = self._build_obs()

        # Rendering
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

        info = {
            "stage": int(self.stage),
            "dist_base": float(dist_b),
            "heading_error": float(heading_err),
            "dist_ee": float(dist_ee),
            "base_reached": bool(base_reached),
            "arm_reached": bool(arm_reached),
            "reached": bool(reached),
            "collision": bool(collided),

            # safety / sensors
            "lidar_min_m": float(lidar_min_m),
            "joint_prox_min": float(np.min(joint_prox)),
            "safety_interventions": int(self.safety_interventions),

            # reward components for debugging
            "prog_base": float(prog_b),
            "prog_ee": float(prog_ee),
            "time_pen": float(self.step_time_penalty),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
