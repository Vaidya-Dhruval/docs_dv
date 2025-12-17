# PPO Hyperparameter Configuration Documentation

This document provides a technical breakdown of the configuration parameters used in the `rl_tidybot_config_pt.py` script for training a Tidybot robot using Proximal Policy Optimization (PPO).

## 1. Parallel Environment Setup

*   **`N_ENVS = 16`**
    *   **Description:** The number of parallel environment instances (workers) running simultaneously.
    *   **Function:** Instead of collecting data from a single simulation, the agent collects observations and rewards from 16 independent simulations in parallel.
    *   **System Impact:** This directly dictates the batch dimension of the input tensor. If `N_ENVS` is 16, the input tensor shape will be `(16, obs_dim)`, which corresponds to the `mat1` dimension in common linear layer errors.
    *   **Trade-off:** Higher values increase data collection throughput (wall-clock speed) and reduce sample correlation but increase CPU and RAM usage.

*   **`BASE_SEED = 123`**
    *   **Description:** The initial seed for pseudo-random number generators (Python, NumPy, PyTorch).
    *   **Function:** Ensures deterministic behavior. Setting a fixed seed guarantees that the environment initialization (e.g., robot starting position, obstacle layout) and network weight initialization are identical across runs.
    *   **Use Case:** Essential for debugging and comparing different algorithm modifications fairly.

## 2. Rollout Buffer (Data Collection)

*   **`N_STEPS = 12288`**
    *   **Description:** The total number of steps collected across *all* environments before a generic update is triggered.
    *   **Calculation:** The number of steps per individual environment is `N_STEPS / N_ENVS`.
        *   `12288 / 16 = 768` steps per environment.
    *   **Implication:** The agent acts for 768 time steps in each world. This defines the "horizon" of experience. It must be sufficient for the agent to potentially complete a task or observe significant rewards (e.g., reaching a goal).

*   **`BATCH_SIZE = 2048`**
    *   **Description:** The size of the mini-batches used during the gradient update phase.
    *   **Function:** The total collected buffer (12,288 steps) is shuffled and split into smaller chunks (minibatches) of 2,048 samples.
    *   **Impact:**
        *   **Too Small:** Updates become noisy and unstable.
        *   **Too Large:** Updates become computationally expensive and may converge to sharp minima. 2048 is a standard value for continuous control tasks.

## 3. PPO Hyperparameters (The Learning Algorithm)

*   **`GAMMA = 0.99` (Discount Factor)**
    *   **Description:** A weighting factor for future rewards in the return calculation ($G_t = R_{t} + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots$).
    *   **Function:** Determines the agent's "horizon" of foresight.
    *   **Value 0.99:** Represents a long-term focus. The agent values a reward received 100 steps in the future almost as much as an immediate one. This is critical for navigation tasks where the sparse "goal reached" reward occurs only after many steps.

*   **`GAE_LAMBDA = 0.95`**
    *   **Description:** The smoothing parameter for Generalized Advantage Estimation (GAE).
    *   **Function:** Controls the trade-off between bias and variance in the advantage estimate.
    *   **Value 0.95:** A widely accepted standard for continuous control (MuJoCo/robotics), providing a balanced estimate that stabilizes training.

*   **`CLIP_RANGE = 0.2`**
    *   **Description:** The hyperparameter $\epsilon$ in the PPO clipped objective function.
    *   **Function:** Limits (clips) the change in the policy probability ratio $r_t(\theta)$ to the range $[1 - \epsilon, 1 + \epsilon]$.
    *   **Purpose:** Prevents "catastrophic forgetting" by ensuring that a single update does not change the policy too drastically, keeping the new policy "proximal" to the old one.

*   **`ENT_COEF = 0.0` (Entropy Coefficient)**
    *   **Description:** The weight of the entropy bonus term in the loss function.
    *   **Function:** Encourages exploration by penalizing the certainty of the policy distribution (rewarding randomness).
    *   **Value 0.0:** Indicates no explicit bonus for randomness. The agent relies solely on the stochasticity of the Gaussian policy action sampling for exploration. If the agent converges prematurely to a suboptimal strategy, increasing this value (e.g., to 0.01) can help.

*   **`VF_COEF = 0.5` (Value Function Coefficient)**
    *   **Description:** A scaling factor for the value function loss ($L^{VF}$) in the total loss equation.
    *   **Function:** Balances the importance of training the Critic (value predictor) relative to the Actor (policy).
    *   **Value 0.5:** A standard default that ensures the value function is learned effectively without overpowering the policy gradient signal.

*   **`N_EPOCHS = 10`**
    *   **Description:** The number of passes over the collected buffer (12,288 steps) during one update phase.
    *   **Function:** The agent re-uses the collected experience 10 times to refine its policy.
    *   **Impact:** Improves sample efficiency (learning more from less data) but increases the computational time per update.

## 4. Training Logistics

*   **`TOTAL_TIMESTEPS = 8_000_000`**
    *   **Description:** The total number of environment interactions to perform before terminating the training script.
    *   **Scale:** 8 million steps is a substantial training run, typical for learning robust navigation policies in complex environments.

*   **`CHECKPOINT_INTERVAL_TIMESTEPS = 500_000`**
    *   **Description:** The frequency (in simulation steps) at which the model weights are saved to disk.
    *   **Purpose:** Facilitates crash recovery and allows for the evaluation of intermediate policies to monitor progress over time.

*   **`EXPERIMENT_TAG` & `RUN_ID`**
    *   **Description:** Identifiers for experiment management.
    *   **Function:** Used to construct unique directory paths for logging and saving models, preventing the overwriting of previous experiment data.