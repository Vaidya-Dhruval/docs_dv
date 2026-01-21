# tidybot_nav_env_v10.py_____ this file is before the per implimentaion 
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_conj(q):
    # q = [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1, q2):
    # Hamilton product, both [w,x,y,z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_angle(q_err):
    # q_err normalized, angle = 2*acos(w)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    return 2.0 * math.acos(w)


class TidybotNavEnvV10(gym.Env):
    """
    V10: Stage-based Base + Arm with real sensors (wrist RGB + lidar).
    - Stage 0: base -> base_goal_site (arm held)
    - Stage 1: ee (pinch_site) -> arm_goal_site (base held)
    - Done: base pos+heading AND arm pos+orientation

    Observation: Dict
      - "image": uint8 (3, H, W) from camera "wrist"
      - "proprio": float32 vector (base errors, ee errors, stage, vels, lidar, arm joints)

    Action: float32 (10,)
      [vx_body, vy_body, wz, dq1..dq7]
    """

    metadata = {"render_modes": ["human", None]}

    BASE_GOAL_SITE = "base_goal_site"
    ARM_GOAL_SITE = "arm_goal_site"
    EE_SITE = "pinch_site"
    CAM_NAME = "wrist"

    # lidar names same as v1-9 (you confirmed)
    LIDAR_SENSOR_NAMES = [
        "rfA_000", "rfA_045", "rfA_090", "rfA_135", "rfA_180",
        "rfB_000", "rfB_090", "rfB_180",
    ]

    def __init__(
        self,
        xml_path="tidybot.xml",
        render_mode=None,
        dt=0.05,
        max_steps=600,
        cam_w=160,
        cam_h=120,
        lidar_max_range=8.0,
        max_vx=0.12,
        max_vy=0.08,
        max_wz=0.25,
        arm_max_dq=(0.25, 0.25, 0.25, 0.35, 0.45, 0.45, 0.45),
        base_reach_radius=0.35,
        base_heading_tol=0.6,
        arm_reach_pos_radius=0.08,
        arm_reach_ori_tol=0.35,
    ):
        super().__init__()
        self.xml_path = xml_path
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.step_count = 0

        self.xy_limit = 5.0

        # ---------------- Base joints ----------------
        jid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
        jid_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y")
        jid_th = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_th")
        if min(jid_x, jid_y, jid_th) < 0:
            raise RuntimeError("Missing base joints joint_x/joint_y/joint_th")

        self.qx = self.model.jnt_qposadr[jid_x]
        self.qy = self.model.jnt_qposadr[jid_y]
        self.qth = self.model.jnt_qposadr[jid_th]

        # Base actuators (you showed these exist)
        self.aid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_x")
        self.aid_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_y")
        self.aid_th = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_th")
        if min(self.aid_x, self.aid_y, self.aid_th) < 0:
            raise RuntimeError("Missing base actuators joint_x/joint_y/joint_th")

        # Base limits
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)

        # ---------------- Arm joints + actuators ----------------
        self.arm_joint_names = [f"joint_{i}" for i in range(1, 8)]
        self.arm_act_ids = []
        self.arm_qadr = []
        self.arm_range = []
        for jn in self.arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            if jid < 0 or aid < 0:
                raise RuntimeError(f"Missing arm joint/actuator named '{jn}'")
            self.arm_act_ids.append(aid)
            self.arm_qadr.append(self.model.jnt_qposadr[jid])
            self.arm_range.append(self.model.jnt_range[jid].copy())

        self.arm_act_ids = np.array(self.arm_act_ids, dtype=np.int32)
        self.arm_qadr = np.array(self.arm_qadr, dtype=np.int32)
        self.arm_range = np.array(self.arm_range, dtype=np.float32)

        self.arm_max_dq = np.array(arm_max_dq, dtype=np.float32)
        self.arm_target = np.zeros(7, dtype=np.float32)

        # ---------------- Sites ----------------
        self.sid_base_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.BASE_GOAL_SITE)
        self.sid_arm_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ARM_GOAL_SITE)
        self.sid_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE)
        if min(self.sid_base_goal, self.sid_arm_goal, self.sid_ee) < 0:
            raise RuntimeError(
                f"Missing sites. Need: {self.BASE_GOAL_SITE}, {self.ARM_GOAL_SITE}, {self.EE_SITE}"
            )

        # thresholds
        self.base_reach_radius = float(base_reach_radius)
        self.base_heading_tol = float(base_heading_tol)
        self.arm_reach_pos_radius = float(arm_reach_pos_radius)
        self.arm_reach_ori_tol = float(arm_reach_ori_tol)

        # ---------------- Lidar sensors ----------------
        self.lidar_indices = []
        for name in self.LIDAR_SENSOR_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise RuntimeError(f"Lidar sensor '{name}' not found in model")
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            if dim != 1:
                raise RuntimeError(f"Lidar sensor '{name}' has dim {dim}, expected 1")
            self.lidar_indices.append(adr)
        self.lidar_indices = np.array(self.lidar_indices, dtype=np.int32)
        self.lidar_max_range = float(lidar_max_range)

        # ---------------- Camera renderer ----------------
        self.cam_w = int(cam_w)
        self.cam_h = int(cam_h)
        self.renderer = mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)

        # ---------------- Internal state ----------------
        self.stage = 0  # 0=base, 1=arm
        self.vcur_base = np.zeros(3, dtype=np.float32)
        self.prev_base = np.zeros(3, dtype=np.float32)
        self.vcur_arm = np.zeros(7, dtype=np.float32)
        self.prev_arm = np.zeros(7, dtype=np.float32)
        self.last_base_dist = None
        self.last_arm_pos_dist = None
        self.last_arm_ori_err = None

        # ---------------- Spaces ----------------
        low = np.concatenate([[-self.max_vx, -self.max_vy, -self.max_wz], -self.arm_max_dq]).astype(np.float32)
        high = np.concatenate([[+self.max_vx, +self.max_vy, +self.max_wz], +self.arm_max_dq]).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # image is channel-first uint8 for SB3 CombinedExtractor
        self.image_space = spaces.Box(low=0, high=255, shape=(3, self.cam_h, self.cam_w), dtype=np.uint8)

        # proprio vector size (fixed)
        # base: dx,dy,dist,heading_err, site_yaw_err, vcur_base(3) => 9
        # ee:   ex,ey,ez, pdx,pdy,pdz, pos_dist, ori_err => 8  (total 17)
        # stage => 1 (18)
        # lidar => 8 (26)
        # arm q (7) => 33
        # arm dq cmd (7) => 40
        self.proprio_dim = 40
        self.proprio_space = spaces.Box(low=-10.0, high=10.0, shape=(self.proprio_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "proprio": self.proprio_space,
        })

    # ---------------- Helpers ----------------
    def _base_pose(self):
        return float(self.data.qpos[self.qx]), float(self.data.qpos[self.qy]), float(self.data.qpos[self.qth])

    def _site_pos(self, sid):
        return self.data.site_xpos[sid].copy()

    def _site_xmat(self, sid):
        # 9 numbers row-major
        return self.data.site_xmat[sid].copy()

    def _site_quat_from_xmat(self, xmat9):
        mat = np.array(xmat9, dtype=np.float64).reshape(3, 3)
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, mat.ravel())
        # normalize
        q /= np.linalg.norm(q) + 1e-9
        return q

    def _base_errors(self):
        goal = self._site_pos(self.sid_base_goal)
        x, y, yaw = self._base_pose()
        dx = float(goal[0] - x)
        dy = float(goal[1] - y)
        dist = float(math.hypot(dx, dy))
        desired = math.atan2(dy, dx)
        heading_err = float(wrap_to_pi(desired - yaw))

        # Also align to base_goal_site orientation yaw (site x-axis yaw)
        xmat = self._site_xmat(self.sid_base_goal)
        x_axis = np.array([xmat[0], xmat[3], xmat[6]], dtype=np.float64)  # (x,y,z) approx
        site_yaw = math.atan2(float(x_axis[1]), float(x_axis[0]))
        site_yaw_err = float(wrap_to_pi(site_yaw - yaw))

        return dx, dy, dist, heading_err, site_yaw_err

    def _arm_errors(self):
        gpos = self._site_pos(self.sid_arm_goal)
        epos = self._site_pos(self.sid_ee)
        pd = gpos - epos
        pos_dist = float(np.linalg.norm(pd))

        # orientation error: angle between goal site and ee site frames
        qg = self._site_quat_from_xmat(self._site_xmat(self.sid_arm_goal))
        qe = self._site_quat_from_xmat(self._site_xmat(self.sid_ee))
        q_err = quat_mul(qg, quat_conj(qe))
        q_err /= np.linalg.norm(q_err) + 1e-9
        ori_err = float(quat_angle(q_err))

        return epos, pd, pos_dist, ori_err

    def _lidar(self):
        vals = []
        for adr in self.lidar_indices:
            r = float(self.data.sensordata[adr])
            if r < 0:
                r = self.lidar_max_range
            r = min(r, self.lidar_max_range)
            vals.append(r / self.lidar_max_range)  # 1=free, 0=close
        return np.array(vals, dtype=np.float32)

    def _render_wrist(self):
        self.renderer.update_scene(self.data, camera=self.CAM_NAME)
        img = self.renderer.render()  # (H,W,3) uint8
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)
        return img

    def _collision(self):
        for c in self.data.contact:
            if c.dist < 0:
                return True
        return False

    def _obs(self):
        # image
        image = self._render_wrist()

        # proprio
        dx, dy, dist, heading_err, site_yaw_err = self._base_errors()
        epos, pd, pos_dist, ori_err = self._arm_errors()
        lidar = self._lidar()

        arm_q = self.data.qpos[self.arm_qadr].astype(np.float32)
        # we include last filtered command as "vcur_arm" (not true qvel, but useful for smoothness)
        vcur_arm = self.vcur_arm.astype(np.float32)

        proprio = np.zeros((self.proprio_dim,), dtype=np.float32)
        k = 0
        proprio[k:k+5] = [dx, dy, dist, heading_err, site_yaw_err]; k += 5
        proprio[k:k+3] = self.vcur_base; k += 3
        proprio[k:k+3] = epos.astype(np.float32); k += 3
        proprio[k:k+3] = pd.astype(np.float32); k += 3
        proprio[k:k+2] = [pos_dist, ori_err]; k += 2
        proprio[k] = float(self.stage); k += 1
        proprio[k:k+len(lidar)] = lidar; k += len(lidar)
        proprio[k:k+7] = arm_q; k += 7
        proprio[k:k+7] = vcur_arm; k += 7

        return {"image": image, "proprio": proprio}

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Base spawn
        self.data.qpos[self.qx] = 0.0
        self.data.qpos[self.qy] = 0.0
        self.data.qpos[self.qth] = 0.0

        # Set base ctrl to match
        self.data.ctrl[self.aid_x] = float(self.data.qpos[self.qx])
        self.data.ctrl[self.aid_y] = float(self.data.qpos[self.qy])
        self.data.ctrl[self.aid_th] = float(self.data.qpos[self.qth])

        mujoco.mj_forward(self.model, self.data)

        # Hold arm at current
        cur = self.data.qpos[self.arm_qadr].astype(np.float32)
        self.arm_target[:] = cur
        for i, aid in enumerate(self.arm_act_ids):
            self.data.ctrl[aid] = float(self.arm_target[i])

        self.step_count = 0
        self.stage = 0

        self.vcur_base[:] = 0
        self.prev_base[:] = 0
        self.vcur_arm[:] = 0
        self.prev_arm[:] = 0

        # init distances
        _, _, dist_b, _, _ = self._base_errors()
        _, _, pos_dist, ori_err = self._arm_errors()
        self.last_base_dist = float(dist_b)
        self.last_arm_pos_dist = float(pos_dist)
        self.last_arm_ori_err = float(ori_err)

        obs = self._obs()
        info = {
            "stage": int(self.stage),
            "dist_base": float(dist_b),
            "arm_pos_dist": float(pos_dist),
            "arm_ori_err": float(ori_err),
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

        # Smooth filtering
        tau = 0.25
        alpha = self.dt / (tau + self.dt)

        self.vcur_base = (1 - alpha) * self.vcur_base + alpha * cmd_base
        self.vcur_arm = (1 - alpha) * self.vcur_arm + alpha * cmd_arm

        accel_base = self.vcur_base - self.prev_base
        accel_arm = self.vcur_arm - self.prev_arm
        self.prev_base[:] = self.vcur_base
        self.prev_arm[:] = self.vcur_arm

        # ---------- Apply base kinematics ----------
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
        self.data.qpos[self.qth] = yaw

        # keep base actuators consistent
        self.data.ctrl[self.aid_x] = float(x)
        self.data.ctrl[self.aid_y] = float(y)
        self.data.ctrl[self.aid_th] = float(yaw)

        # ---------- Apply arm position targets ----------
        self.arm_target[:] = self.arm_target + self.vcur_arm * self.dt
        self.arm_target[:] = np.clip(self.arm_target, self.arm_range[:, 0], self.arm_range[:, 1])
        for i, aid in enumerate(self.arm_act_ids):
            self.data.ctrl[aid] = float(self.arm_target[i])

        # Step physics
        mujoco.mj_step(self.model, self.data)

        # ---------- Compute errors ----------
        dx, dy, dist_b, heading_err, site_yaw_err = self._base_errors()
        _, _, arm_pos_dist, arm_ori_err = self._arm_errors()

        # Progress
        prog_b = float(self.last_base_dist - dist_b)
        prog_ap = float(self.last_arm_pos_dist - arm_pos_dist)
        prog_ao = float(self.last_arm_ori_err - arm_ori_err)

        self.last_base_dist = float(dist_b)
        self.last_arm_pos_dist = float(arm_pos_dist)
        self.last_arm_ori_err = float(arm_ori_err)

        collided = self._collision()

        # Stage completion
        base_reached = (dist_b < 0.35) and (abs(heading_err) < 0.6)
        if self.stage == 0 and base_reached:
            self.stage = 1

        arm_reached = (arm_pos_dist < 0.08) and (arm_ori_err < 0.35)
        reached = bool(base_reached and arm_reached)

        # ---------- Reward ----------
        reward = 0.0

        # Weights (stable, not “elephant trunk”)
        if self.stage == 0:
            reward += 2.0 * prog_b
            reward += -0.2 * dist_b
            reward += -0.05 * abs(heading_err)
            reward += -0.02 * abs(site_yaw_err)
            reward += -0.05 * float(np.dot(self.vcur_base, self.vcur_base))
            reward += -0.02 * float(np.dot(accel_base, accel_base))
        else:
            reward += 3.0 * prog_ap
            reward += 0.5 * prog_ao   # orientation improvement matters
            reward += -0.4 * arm_pos_dist
            reward += -0.15 * arm_ori_err
            reward += -0.02 * float(np.dot(self.vcur_arm, self.vcur_arm))
            reward += -0.01 * float(np.dot(accel_arm, accel_arm))

        if collided:
            reward += -8.0

        if reached:
            reward += +40.0

        timeout = self.step_count >= self.max_steps
        terminated = reached
        truncated = bool(timeout and not reached)
        if truncated:
            reward += -6.0

        obs = self._obs()
        info = {
            "stage": int(self.stage),
            "dist_base": float(dist_b),
            "heading_error": float(heading_err),
            "arm_pos_dist": float(arm_pos_dist),
            "arm_ori_err": float(arm_ori_err),
            "base_reached": bool(base_reached),
            "arm_reached": bool(arm_reached),
            "reached": bool(reached),
            "collision": bool(collided),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
