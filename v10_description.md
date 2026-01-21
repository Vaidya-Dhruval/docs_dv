# TidybotNavEnvV10 — Static- (128×128, CPU)


- **Stage 0:** base reaches `base_goal_site` (position + heading)
- **Stage 1:** arm reaches `arm_goal_site` (position) **and** points down (orientation)
- **Reached=True:** both stage successes satisfied

Static-first: **mocap obstacles remain static** (no scripted motion). This is the recommended baseline to achieve reliable successes before introducing dynamic obstacles.

## Files

- `tidybot_nav_env_v10.py` — environment (Multi-Modal obs, staged reward/termination)
- `v10_static_config.py` — parameters + PPO hyperparams + curriculum schedule
- `train_v10_static_first.py` — 3-phase curriculum training with progress bars
- `eval_v10_window.py` — evaluation with MuJoCo viewer + optional wrist image window (if `opencv-python` installed)

## Key XML assumptions

Your MuJoCo XML must contain:

- Sites:
  - `base_goal_site`
  - `arm_goal_site`
  - `pinch_site`
- Camera:
  - `wrist`
- Actuators:
  - base: `joint_x`, `joint_y`, `joint_th`
  - arm: `joint_1..joint_7`
- Rangefinder sensors (subset used by env):
  - `rfA_000 rfA_045 rfA_090 rfA_135 rfA_180 rfA_225 rfA_270 rfA_315`
  - `rfB_000 rfB_045 rfB_090 rfB_135 rfB_180 rfB_225 rfB_270 rfB_315`

If your XML uses different names, update `lidar_sensor_names` in `tidybot_nav_env_v10.py`.

## Orientation constraint (Option A)

Orientation success uses:

- `ez = z-axis of pinch_site frame`
- target direction = world down `[0,0,-1]`
- `ori_err = acos(clip(dot(ez, down), -1, 1))`

Arm reached condition:

- `dist_ee < arm_reach_radius` **and**
- `ori_err < arm_ori_tol`

## Training

```bash
python3 train_v10_static_first.py
```

This runs:

1) **base-only** (stage_fixed=0)  
2) **arm-only** (stage_fixed=1, base is spawned near base_goal)  
3) **full** (stage_fixed=None, base->arm transition enabled)

Models are saved in:

`./tb_tidybot_nav/v10/<RUN_NAME>/run_<timestamp>/`

## Evaluation

```bash
python3 eval_v10_window.py --model /path/to/ppo_v10_full.zip --episodes 5
```

- MuJoCo passive viewer opens (window).
- If `opencv-python` is installed, a second window shows the wrist RGB feed.

## Why this is stable

- **No premature freezing:** safety layer scales commands but never hard-stops.
- **Progress rewards:** both stages use `(prev_dist - dist)` as the main reward term.
- **Curriculum:** arm training is not blocked by base navigation errors.
- **CNN safety:** camera is provided as **CHW uint8**, compatible with SB3 NatureCNN.