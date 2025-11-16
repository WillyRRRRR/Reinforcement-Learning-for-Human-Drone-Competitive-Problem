import torch
import torch.nn as nn

class DriverActor(nn.Module):
    """Policy network for the driver agent (with mean-field input).
    Inspired by the MFCPPO design, where the mean-field action is also fed into the network."""
    def __init__(self, obs_size, mean_action_size, action_size): 
        # Input: observation and mean-field action; Output: action probability distribution
        super(DriverActor, self).__init__()
        self.obs_size = obs_size          # Dimension of partial observation
        self.mean_action_size = mean_action_size  # Mean-field action dimension (specific to Mean-Field PPO)
        self.action_size = action_size    # Dimension of output action space
        
        self.net = nn.Sequential(
            nn.Linear(obs_size + mean_action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs, mean_action):
        x = torch.cat([obs, mean_action], dim=-1)  # Fix concatenation issue
        return self.net(x)  # Output: probability distribution over driver actions


class DriverCritic(nn.Module):  
    """Critic network for the driver agent (with mean-field input).
    Takes partial observation and mean-field action to estimate state value V(s)."""
    def __init__(self, obs_size, mean_action_size):
        super(DriverCritic, self).__init__()
        input_size = obs_size + mean_action_size  
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, obs, mean_action):
        """
        obs: [B, obs_size]              - Driver's observation
        mean_action: [B] or [B, 1]      - Proportion of the population choosing this action
        """
        # Concatenate observation and mean-field action
        x = torch.cat([obs, mean_action], dim=-1)  # [B, obs_size + mean_action_size]
        return self.net(x)  # [B, 1]


class DriverMemory:
    def __init__(self):  
        self.observations = []         # Current observations
        self.mean_actions = []         # Mean-field actions
        self.actions = []              # Chosen actions
        self.rewards = []              # Received rewards
        self.obs_values = []           # Value estimates for current observations
        self.logprobs = []             # Log-probabilities of chosen actions
        self.next_observations = []    # Next-step observations
        self.next_mean_actions = []    # Next-step mean-field actions
        self.is_dones = []             # Episode termination flags
        self.next_obs_values = []      # Value estimates for next observations

    def clear_memory(self): 
        del self.observations[:]         # Clear observations
        del self.mean_actions[:]         # Clear mean-field actions
        del self.actions[:]              # Clear actions
        del self.rewards[:]              # Clear rewards
        del self.obs_values[:]           # Clear observation values
        del self.logprobs[:]             # Clear log-probabilities
        del self.next_observations[:]    # Clear next observations
        del self.next_mean_actions[:]    # Clear next mean-field actions
        del self.is_dones[:]             # Clear termination flags
        del self.next_obs_values[:]      # Clear next observation values

    def push_t(self, obs, mean_action, action, reward, obs_value, logprob): 
        # Store data available at the current time step
        self.observations.append(obs)
        self.mean_actions.append(mean_action)
        self.actions.append(action)
        self.rewards.append(reward)
        self.obs_values.append(obs_value)
        self.logprobs.append(logprob)

    def push_tt(self, next_obs, next_mean_action, done, next_obs_value): 
        # Store data only available after transitioning to the next time step
        self.next_observations.append(next_obs)
        self.next_mean_actions.append(next_mean_action)
        self.is_dones.append(done)
        self.next_obs_values.append(next_obs_value)

    def sample(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.stack(self.observations).to(device),          
            torch.stack(self.mean_actions).to(device),          
            torch.stack(self.actions).to(device),               
            torch.tensor(self.rewards, dtype=torch.float32).to(device),
            torch.stack(self.obs_values).to(device),         
            torch.stack(self.logprobs).to(device),            
            torch.stack(self.next_observations).to(device),  
            torch.stack(self.next_mean_actions).to(device),    
            torch.tensor(self.is_dones, dtype=torch.float32).to(device), 
            torch.stack(self.next_obs_values).to(device)    
        )


