# RL² for Parameter Identification with Unknown Disturbances

This repository demonstrates the use of RL² (Reinforcement Learning with Recurrent Mechanisms) for **parameter identification** in systems with **unknown disturbances**. The goal is to utilize RL²'s meta-learning capabilities to identify parameters of a dynamic system while accounting for stochastic or deterministic disturbances that impact the process.

---

## **Key Features**
- **Meta-Reinforcement Learning:**
  - RL² uses a recurrent policy that processes observation histories to adapt to new tasks during inference.
  - This makes it ideal for environments with variability or incomplete prior knowledge.

- **Parameter Identification with Unknown Disturbances:**
  - Our framework estimates system parameters using simulated environments with randomized, unknown disturbances.
  - The methodology assumes limited prior knowledge of the disturbance characteristics, challenging the agent to infer and adapt online.

- **Flexible Environment Support:**
  - Includes support for environments like Lunar Lander with modified dynamics and noise injection.
  - Easily extensible to other OpenAI Gym-like environments or custom domains.

---

## **Workflow**

The project consists of the following pipeline:

1. **Environment Setup:**
   - Define a dynamic environment with configurable parameters and disturbance models (e.g., random noise, time-varying effects).
   - Example: Lunar Lander with randomized wind forces and gravity perturbations.

2. **RL² Training:**
   - Train a recurrent policy using Proximal Policy Optimization (PPO) or another RL algorithm.
   - The policy learns to identify task parameters by leveraging historical observations.

3. **Evaluation:**
   - Evaluate the trained policy in environments with unseen disturbances.
   - Compare performance with traditional methods (e.g., PPO).

---

## **Repository Structure**

```plaintext
.
|-- rl2/
|   |-- agents/               # RL² agent implementations (PPO, RNN policies)
|   |-- preprocessing/        # Modules for input preprocessing (e.g., one-hot encoding)
|   |-- environments/         # Custom and adapted environments
|   |-- utils/                # Helper functions for training and evaluation
|
|-- experiments/
|   |-- lunar_lander/         # Configurations and scripts for Lunar Lander tasks
|   |-- results/              # Logs and trained model checkpoints
|
|-- tests/                    # Unit tests for all major modules
|-- README.md                 # Project documentation
```

---

## **Installation**

```bash
# Clone the repository
$ git clone https://github.com/mateusbsal4/tum-adlr-ws25-01.git
$ cd tum-adlr-ws25-01

# Install required dependencies

```

---

## **Quickstart**

### Train the RL² Agent on Lunar Lander


---

## **Key Results**

1. **Lunar Lander Experiments:**
   
2. **Generalization:**

3. **Efficiency:**

---

## **References**
- RL²: Wang, Jane X., et al. "Learning to reinforcement learn." *arXiv preprint arXiv:1611.05763* (2016).
- OpenAI Gym: Brockman, Greg, et al. "OpenAI Gym." *arXiv preprint arXiv:1606.01540* (2016).

---
