# Reinforcement Learning Courier-Drone Project

This repository contains a simulation environment and core components for a reinforcement learning-based courier (delivery person) agent. The project is organized around five main Python files, each responsible for a key part of the system:

## Project Structure

```
.
├── data_process.py          # Data processing script
├── dual_head_network.py     # Dual-head neural network
├── courier_agent.py         # Courier agent logic
├── train_ppo.py             # PPO (Proximal Policy Optimization) training script
└── virtual_env.py           # Virtual environment simulation
```

## File Descriptions

- **data_process.py**  
  Handles loading, cleaning, and preprocessing raw data, transforming it into a format suitable for reinforcement learning experiments.

- **dual_head_network.py**  
  Defines the dual-head neural network architecture, responsible for both policy and value predictions within the RL setup.

- **courier_agent.py**  
  Implements the courier (delivery agent) logic, including state observation, action selection, and interaction with the environment.

- **train_ppo.py**  
  Contains the PPO training loop, including loss calculation, model updates, and checkpointing.

- **virtual_env.py**  
  Builds the simulated delivery environment, including order generation, task assignment, and feedback mechanisms for the agent.

## Getting Started

1. Preprocess your data using `data_process.py`.
2. Set up the simulation environment with `virtual_env.py`.
3. Start training the agent by running `train_ppo.py`. The agent and network definitions are located in `courier_agent.py` and `dual_head_network.py`, respectively.
4. Review and customize parameters as needed in each script.

## Requirements

- Python 3.8+
- numpy
- torch
- (Other dependencies; see `requirements.txt` if available)

---

For more details on each module or experiment results, please refer to the comments and documentation within each file.
