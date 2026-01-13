# Reinforcement Learning–Based Mobile Manipulation for Industrial Machine Tending  
*A Progressive Experimental Study Using MuJoCo and PPO*

---

## Abstract

This work presents an extensive experimental study on reinforcement learning (RL)–based control of a mobile manipulator in a simulated industrial laboratory environment. The robotic system under study consists of a mobile base combined with a 7-DoF manipulator, modeled in MuJoCo and inspired by a Stanford Tidybot–like platform. The primary objective is machine tending: autonomously navigating the mobile base to a working table and subsequently performing a manipulation operation with the robotic arm.

The study documents a progressive development process from simple base-only navigation (v1–v3) to staged base–arm coordination (v8–v9), and finally to perception-driven control using convolutional neural networks (v10). Particular emphasis is placed on practical challenges encountered during training, including unstable policies, reward misalignment, actuator conflicts, observation–action mismatches, collision handling, and GPU memory constraints.

Rather than presenting only successful results, this report focuses heavily on failure cases and debugging cycles, which played a crucial role in shaping the final architecture. The findings suggest that structured task decomposition, staged learning, and perception-aware policies are essential prerequisites for deploying reinforcement learning–based mobile manipulators in industrial environments.

---

## 1. Introduction

Industrial automation increasingly demands robotic systems that are flexible, adaptive, and capable of operating in semi-structured environments. Traditional industrial robots rely on carefully engineered pipelines consisting of perception modules, global and local planners, inverse kinematics solvers, and task schedulers. While such systems are reliable, they often require significant manual tuning and can be brittle when the environment changes.

Reinforcement learning offers an alternative paradigm in which control policies are learned directly through interaction with the environment. In principle, an RL agent can adapt to variability in geometry, dynamics, and task specifications without explicit reprogramming. However, applying RL to **mobile manipulation** remains particularly challenging due to the high dimensionality of the state and action spaces, the coupling between base and arm dynamics, and strict safety constraints.

This project explores whether a single RL-based framework can be progressively extended to perform an industrially relevant machine tending task, and what architectural decisions are necessary to achieve stable and interpretable learning behavior.

---

## 2. Simulation Environment and Robot Model

### 2.1 MuJoCo Simulation Framework

All experiments were conducted using the MuJoCo physics engine due to its accurate contact modeling, deterministic stepping, and efficient simulation of articulated systems. The simulated environment represents a simplified industrial laboratory and includes:

- Static tables and working surfaces  
- Fixed walls and boundaries  
- Optional dynamic obstacles (moving boxes, humanoid dummy)  
- A mobile manipulator robot  

The use of a deterministic simulator allowed the study to focus on learning-related instabilities rather than stochastic simulation noise.

---

### 2.2 Robot Kinematics and Actuation

The robot model consists of two tightly coupled subsystems:

**Mobile base**
- Planar motion with three degrees of freedom: x, y, and yaw  
- Implemented using two slide joints and one rotational joint  
- Controlled via body-frame velocity commands integrated kinematically  

**Robotic arm**
- Seven revolute joints (`joint_1` to `joint_7`)  
- Position-controlled actuators with joint limits enforced  
- End-effector reference defined via a `pinch_site`  

This combination results in a high-dimensional control problem that is representative of real industrial mobile manipulators.

---

## 3. Task Definition: Machine Tending

The machine tending task investigated in this work consists of two logically distinct but physically coupled objectives:

1. **Base positioning**  
   The mobile base must navigate toward a working table and stop within a predefined tolerance region.

2. **Arm manipulation**  
   Once the base is positioned, the robotic arm must reach a feasible target location on the table surface.

A key insight that emerged early in the project is that **a single shared goal for both base and arm is physically infeasible**. The base operates in the horizontal plane, while the arm must reach vertically and laterally within its kinematic workspace. This realization motivated the staged objective formulation introduced in later versions.

---

## 4. Progressive Development of Environment Versions

### 4.1 Early Versions (v1–v3): Base-Only Navigation

The initial versions focused exclusively on mobile base navigation.

**Observation space**
- Base position (x, y)  
- Heading (yaw)  
- Relative goal displacement (dx, dy)  
- Euclidean distance to the goal  

**Action space**
- Body-frame velocities (vx, vy, ω)

**Observed behavior**
- Extremely fast and unrealistic motion  
- Oscillations near the goal  
- Failure to settle into a stable stopping configuration  

**Lessons learned**
- Velocity and acceleration smoothing are essential  
- Pure distance-based rewards are insufficient  
- Progress-based shaping improves convergence  

---

### 4.2 Intermediate Versions (v4–v5): Reward Shaping and Sensors

Subsequent versions introduced:

- First-order action smoothing  
- Explicit collision penalties using MuJoCo contact data  
- Coarse lidar signals in the observation space  

Despite these additions, the robot frequently stopped near table legs rather than the intended goal. This behavior revealed that local minima in the reward landscape can dominate learning when goals are poorly defined.

---

### 4.3 Naïve Base + Arm Integration (v6–v7)

In v6 and v7, the arm was directly added to the action and observation spaces, allowing simultaneous control of base and arm.

**Failure modes**
- Base freezing shortly after episode start  
- Arm exhibiting large, erratic motions at spawn  
- Episodes terminating with `reached = false` despite visually correct base positioning  

**Root cause analysis**
- Conflicting gradients from base and arm objectives  
- Arm goal positions unreachable from the base pose  
- Credit assignment ambiguity between subsystems  

These failures highlighted the limitations of naïve end-to-end learning for mobile manipulation.

---

### 4.4 Staged Objective Learning (v8–v9)

To address the observed instabilities, a staged learning approach was introduced:

- **Stage 0**:  
  - Only base actions are enabled  
  - Arm is held fixed  
  - Objective: reach base goal site  

- **Stage 1**:  
  - Base is frozen  
  - Arm actions are enabled  
  - Objective: reach arm goal site  

This architectural change significantly improved training stability and interpretability.

---

## 5. v10: CNN-Based Perception Policy

### 5.1 Motivation for Visual Perception

All versions up to v9 relied on low-dimensional state vectors derived from simulator ground truth. While effective in simulation, such representations are not realistic for real-world deployment, where precise state estimation is imperfect.

Version v10 therefore introduces **visual perception** via an onboard camera and a convolutional neural network (CNN) policy.

---

### 5.2 CNN Observation Design

The CNN observation consists of:

- RGB images from a wrist-mounted or arm-mounted camera  
- Downsampled resolution for computational efficiency  
- Optional temporal stacking  

The CNN processes spatial information such as table edges, objects, and relative positioning that are difficult to encode manually.

---

### 5.3 CNN Training Challenges

Significant challenges were encountered when training CNN-based policies:

- GPU out-of-memory (OOM) errors due to large rollout buffers  
- High memory consumption from `N_ENVS × N_STEPS`  
- Fragmentation of CUDA memory during PPO updates  

**Mitigation strategies**
- Reduced batch sizes and number of epochs  
- CPU-based overnight training for reliability  
- Automatic fallback from CUDA to CPU execution  

---

## 6. Reward Function Design

The final reward structure is explicitly stage-dependent.

### 6.1 Base Stage Reward

\[
R = w_p \Delta d - w_d d - w_h |\theta| - w_u \|u\|^2 - w_a \|\dot{u}\|^2
\]

Where:
- \( \Delta d \) is progress toward the base goal  
- \( d \) is distance to the base goal  
- \( \theta \) is heading error  

---

### 6.2 Arm Stage Reward

\[
R = w_p \Delta d_{ee} - w_d d_{ee} - w_u \|q\|^2
\]

This formulation prioritizes smooth, precise end-effector motion while penalizing excessive joint activity.

---

## 7. Failure Analysis

One of the main contributions of this work is the systematic documentation of failure cases:

- **Base stopping near table legs** due to local reward minima  
- **Arm flailing at spawn** caused by unreachable goals  
- **Training collapse** when base and arm objectives conflicted  
- **CNN OOM crashes** caused by rollout buffer scaling  

Each failure directly informed subsequent design changes.

---

## 8. Industrial Readiness Assessment

While v10 represents a significant research milestone, several steps remain before deployment on real hardware:

- Domain randomization for visual robustness  
- Sensor noise modeling  
- Safety constraints and fallback controllers  
- Integration with classical planning for hybrid control  

Nevertheless, the staged RL framework demonstrated here is structurally compatible with industrial automation requirements.

---

## 9. Future Work

Future research directions include:

- Multi-input CNN policies combining vision and proprioception  
- Hierarchical reinforcement learning  
- Integration of semantic perception  
- Real-world sim-to-real transfer  

---

## 10. Conclusion

This work demonstrates that reinforcement learning for mobile manipulation is not a single algorithmic problem but an engineering process. Through progressive experimentation from v1 to v10, it becomes evident that structured task decomposition, staged control, and perception-aware policies are essential for stability and interpretability.

The final v10 architecture represents a meaningful step toward deployable, learning-based machine tending systems and provides a solid foundation for future industrial research.

---
