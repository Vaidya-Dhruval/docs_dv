# Tidybot V10 Environment  
## Observation, Action, Reward Design (Up to 3-Stage V10)

---

## 1. Task Overview

The V10 environment addresses a **mobile manipulation machine-tending task** using a single PPO policy.

The task is decomposed into **two sequential stages**:

- **Stage 0 – Base Navigation**
  The mobile base navigates to a predefined parking pose near the work table.

- **Stage 1 – Arm Reaching**
  The manipulator moves its end-effector to a target location on the table.

- **Stage 2 – Success (Terminal)**
  Episode terminates when both base and arm goals are satisfied.

Stage switching is handled internally by the environment using geometric thresholds.

---

## 2. Observation Space

V10 uses a **multi-modal observation** composed of vision, proprioception, and range sensing.

### 2.1 Camera Observation (CNN Input)

- Source: Wrist-mounted RGB camera
- Resolution: 128 × 128
- Channels: RGB
- Frame: End-effector frame, pointing downward (Option A)
- Purpose:
  - Visual awareness of table and goal region
  - Industrial realism for machine tending
  - Future extensibility to object-level tasks

The image observation is processed by a CNN feature extractor.

---

### 2.2 State Vector Observation (MLP Input)

A continuous numerical vector containing:

#### Base State
- Base position (x, y)
- Base orientation (yaw)
- Distance to base goal
- Heading error relative to base goal

#### Arm / End-Effector State
- End-effector Cartesian position (x, y, z)
- Distance to arm goal site

#### Velocities
- Base velocities (vx, vy, ω)
- Arm joint velocities (dq1 … dq7)

#### LiDAR
- Normalized rangefinder distances
- Used for obstacle awareness and safety reasoning

#### Task Context
- Current stage indicator (0 or 1)

This state vector is processed by an MLP head.

---

## 3. Action Space

The action space is continuous and shared across stages.

### 3.1 Base Actions (Stage 0 Only)

- vx: forward velocity
- vy: lateral velocity
- ω: angular velocity

These actions are applied only during Stage 0.

---

### 3.2 Arm Actions (Stage 1 Only)

- Joint velocity commands:
  dq1, dq2, dq3, dq4, dq5, dq6, dq7

These actions are applied only during Stage 1.

---

### 3.3 Stage Gating

- During Stage 0: arm actions are ignored
- During Stage 1: base actions are ignored

This gating prevents interference and stabilizes learning.

---

## 4. Reward Design

The reward function is **stage-dependent**, dense, and augmented with terminal bonuses.

---

## 5. Rewards – Stage 0 (Base Navigation)

### Base Progress Reward
Encourages forward progress toward the base goal.

Reward is proportional to the reduction in distance between successive steps.

---

### Base Distance Penalty
Penalizes remaining distance to the base goal to maintain global attraction.

---

### Base Orientation Penalty
Penalizes absolute heading error relative to the goal orientation.

This term enforces correct parking orientation but is sensitive and can dominate early learning.

---

### Control Effort Penalty
Penalizes squared base velocity magnitude to encourage smooth motion.

---

### Acceleration Penalty
Penalizes changes in base velocity to reduce jitter and oscillations.

---

## 6. Rewards – Stage 1 (Arm Reaching)

### End-Effector Progress Reward
Positive reward proportional to reduction in end-effector distance to the arm goal.

---

### End-Effector Distance Penalty
Penalizes absolute distance to encourage precision near the target.

---

### Arm Control Penalty
Penalizes squared joint velocity magnitude to avoid aggressive motions.

---

### Arm Acceleration Penalty
Penalizes joint acceleration to improve smoothness and realism.

---

## 7. Global Penalties (All Stages)

### Time Penalty
A small negative reward applied at every timestep to discourage idling.

---

### Collision Penalty
A large negative reward applied upon any collision for safety enforcement.

---

### Timeout Penalty
Applied when the episode reaches the maximum step limit without success.

---

## 8. Terminal Reward

### Success Bonus
A large positive reward granted only when:
- Base goal is reached (position + orientation)
- Arm goal is reached (end-effector proximity)

This reward defines task completion.

---

## 9. Observed Outcome up to V10 (Static Stage)

- The policy remains in Stage 0
- Base does not reliably reduce distance to goal
- Orientation error remains near π radians
- Episodes terminate due to timeout rather than success

Primary limiting factor is **reward dominance**, where orientation and control penalties suppress exploratory motion.

---

## 10. Summary Statement (Meeting-Ready)

The V10 environment implements an industry-grade observation and action structure.  
Current performance limitations arise from reward interactions in Stage 0 rather than architectural deficiencies.

---
