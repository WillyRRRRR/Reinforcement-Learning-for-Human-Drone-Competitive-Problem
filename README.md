This project references the following article:

Jiaohong Xie, Yang Liu, and Nan Chen. "Two-Sided Deep Reinforcement Learning for Dynamic Mobility-on-Demand Management with Mixed Autonomy." Transportation Science, 2023, 57(4): 1019–1046.

# Reinforcement Learning Courier-Drone Project

This repository contains a simulation environment and core components for a 2 sided reinforcement learning-based algorithm. The project is organized around five main Python files, each responsible for a key part of the system.:

## Project Structure

```
.
├── data_process.py          # Data processing script
├── dnetwork.py              # Dual-head neural network
├── network.py               # Courier agent logic
├── ppo.py                   # PPO (Proximal Policy Optimization) training script base on clip function
└── env.py                   # Virtual environment simulation
```

## File Descriptions

- **data_process.py**  
  Handles loading, cleaning, and preprocessing raw data, transforming it into a format suitable for reinforcement learning experiments.

- **dnetwork.py**  
  Defines the dual-head neural network architecture, responsible for both policy and value predictions within the RL setup.

- **network.py**  
  Implements the courier (delivery agent) logic, including state observation, action selection, and interaction with the environment.

- **ppo2.py**  
  Contains the PPO united training loop, including loss calculation, model updates, and checkpointing.

- **env.py**  
  Builds the simulated delivery environment, including order generation, task assignment, and feedback mechanisms for the agent.

## Getting Started

1. Preprocess your data using `data_process.py`.
2. Set up the simulation environment with `env.py`.
3. Start training the agent by running `ppo2.py`. The agent and network definitions are located in `network.py` and `dnetwork.py`, respectively.
4. Review and customize parameters as needed in each script.

## Requirements

- Python 3.8+
- numpy
- torch with cuda12.6
- (Other dependencies; see `requirements.txt` if available)

---

For more details on each module or experiment results, please refer to the comments and documentation within each file.
