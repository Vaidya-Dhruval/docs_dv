# Tidybot V10: Multi-Modal Reinforcement Learning for Mobile Manipulation  
**Static Environment – Option A (Wrist-Down Observer, 128×128, CPU)**

---

## 1. Introduction

Autonomous machine tending using mobile manipulators requires coordinated navigation, perception, and manipulation under tight spatial constraints. Classical approaches typically rely on modular pipelines involving mapping, localization, motion planning, and visual servoing. While robust, these pipelines demand extensive system integration and tuning.

This work investigates an alternative paradigm: **end-to-end reinforcement learning (RL)** for mobile manipulation using **multi-modal sensory input**, trained entirely in simulation. The objective is not to replace classical planners outright, but to evaluate the feasibility, limitations, and learning dynamics of a unified RL policy capable of sequential base positioning and arm reaching.

The experiments described here focus on **V10**, the most advanced iteration of the training environment developed during this work.

---

## 2. System Overview

### 2.1 Robotic Platform (Simulated)

The simulated system consists of:

- A holonomic mobile base (planar x, y, yaw)
- A 7-DoF Kinova Gen3 robotic arm
- A Robotiq 2F-85 gripper (passive for this study)
- A wrist-mounted RGB camera
- Two 360° LiDAR rings mounted on the base

The simulation is implemented in **MuJoCo**, chosen for its accurate contact modeling and efficient physics stepping.

---

## 3. Task Definition

The machine-tending task is decomposed into two sequential objectives:

1. **Base Navigation**  
   The mobile base must reach a predefined parking location near a work table.

2. **Arm Reaching**  
   Once the base is correctly positioned, the robotic arm must move its end-effector to a target site located above the table surface.

A successful episode is defined as satisfying both conditions within a single rollout.

---

## 4. Observation Space Design

To support multi-modal reasoning, the observation space combines:

### 4.1 Visual Input
- Wrist-mounted RGB camera
- Resolution: **128 × 128**
- Used as an observer view rather than a direct servoing signal

### 4.2 LiDAR Input
- Discrete rangefinder beams sampled from two LiDAR rings
- Encodes obstacle proximity and free-space structure

### 4.3 Low-Dimensional State Vector
Includes:
- Base pose and heading error relative to goal
- End-effector Cartesian error relative to arm target
- Current stage indicator
- Base and arm velocity estimates
- Joint limit proximity indicators

This structured representation allows the policy to reason over geometry, progress, and safety constraints simultaneously.

---

## 5. Action Space

The action space is continuous and structured as:

    [vx, vy, ωz, dq1, dq2, dq3, dq4, dq5, dq6, dq7]
    

- Base velocities are expressed in the robot body frame
- Arm commands represent joint-space velocity targets
- Action gating is applied:
  - During base navigation, arm actions are suppressed
  - During arm reaching, base actions are suppressed

This gating enforces temporal structure without requiring multiple policies.

---

## 6. Policy Architecture

Training uses **PPO with a Multi-Input Policy**:

- **CNN encoder** for RGB input
- **MLP encoder** for state and LiDAR inputs
- Feature fusion followed by shared policy and value heads

This architecture enables the policy to jointly reason over perception and structured state without hand-crafted feature fusion.

---

## 7. Three-Stage Training Strategy

### 7.1 Stage 0 – Base Navigation

**Objective:**  
Minimize distance and heading error to the base goal site.

**Reward components include:**
- Distance progress toward the goal
- Penalty for heading misalignment
- Control smoothness penalties
- Time penalty to encourage efficiency

**Observed behavior:**  
The base learns goal-directed motion and obstacle awareness using LiDAR signals. Near-goal behavior shows increasing sensitivity to reward gradients and safety constraints.

---

### 7.2 Stage 1 – Arm Reaching

**Objective:**  
Move the end-effector to the arm goal site above the table.

**Key characteristics:**
- Wrist camera provides contextual scene information
- End-effector distance drives reward shaping
- Joint-limit proximity penalties encourage feasible postures

**Observed behavior:**  
The arm learns coarse reaching motions and maintains collision-free trajectories in the static environment.

---

### 7.3 Stage 2 – Sequential Integration

**Objective:**  
Execute base navigation followed by arm reaching within a single episode.

**Mechanism:**
- A stage flag switches reward functions and action gating
- Success is defined as:

base_reached AND arm_reached


**Observed behavior:**  
The policy demonstrates partial temporal coordination between navigation and manipulation, highlighting the complexity of long-horizon decision making under shared control parameters.

---

## 8. Reward Shaping and Safety Constraints

Reward shaping plays a central role in stabilizing learning:

- Progress-based rewards dominate early learning
- Control penalties reduce oscillatory behavior
- Safety scaling limits motion near obstacles and joint limits

The interaction between safety constraints and reward gradients significantly influences near-goal behavior and policy smoothness.

---

## 9. Evaluation Setup

Evaluation is performed using a dedicated windowed evaluation script:

- Deterministic policy execution
- Fixed camera resolution
- Real-time rendering (when backend permits)
- Multiple rollouts per checkpoint

Rendering backend stability (GLFW / GLX) was identified as an important practical consideration during evaluation.

---

## 10. Discussion

This study highlights the challenges inherent in **end-to-end RL for mobile manipulation**:

- Sequential objectives require implicit temporal abstraction
- Multi-modal perception must be grounded in task semantics
- Safety mechanisms interact non-trivially with reward optimization
- Long-horizon PPO policies are sensitive to termination and reward thresholds

Despite these challenges, the V10 environment demonstrates that a single unified policy can reason over navigation and manipulation using raw sensory inputs.

---

## 11. Lessons Learned

Key insights from this work include:

- Explicit stage structuring significantly stabilizes learning
- Wrist-mounted cameras are valuable observers but require additional grounding for precise control
- Reward smoothness near goal regions is critical for stable convergence
- Evaluation infrastructure is as important as training itself

---

## 12. Future Directions

Potential extensions of this work include:

- Hierarchical policies with explicit skill decomposition
- Curriculum learning across environment complexity
- Learned visual attention mechanisms
- Integration with classical planners for hybrid control

These directions are left for future investigation.

---

## 13. Conclusion

The V10 implementation represents a comprehensive investigation into multi-modal RL for mobile manipulation. While the task complexity exposes limitations of monolithic policies, the environment and training framework established here provide a strong foundation for future research in learned robot autonomy.

- Din't worked out