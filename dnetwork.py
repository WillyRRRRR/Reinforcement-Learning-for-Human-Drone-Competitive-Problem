import torch
import torch.nn as nn

# Dual-head supervised network file
class TwoHeadActor(nn.Module):
    """Dual-head policy network for a supervised agent"""
    def __init__(self, state_size, action_size_relocation, action_size_cf): 
        # Input: state; Outputs: two types of actions:
        # - Discrete relocation action probability distribution
        # - Commission fee rate distribution across regions
        super(TwoHeadActor, self).__init__()
        self.action_size_relocation = action_size_relocation  # Instance variable declaration
        self.action_size_cf = action_size_cf  # Instance variable declaration
        
        # Shared feature extraction layers
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # Relocation decision head
        self.relocation_head = nn.Sequential(
            nn.Linear(32, action_size_relocation),
            nn.Softmax(dim=-1)
        )
        
        # Commission fee (CF) decision head
        self.cf_head = nn.Sequential(
            nn.Linear(32, action_size_cf),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):  # Forward pass
        shared_features = self.shared_net(state)  # Features from shared layers
        relocation_probs = self.relocation_head(shared_features)  # Probabilities for all relocation actions; actual action obtained via sampling
        cf_probs = self.cf_head(shared_features)  # Probabilities for all commission fee rates; actual rate obtained via sampling
        return relocation_probs, cf_probs  # Return probability distributions for relocation and commission fee


class TwoHeadCritic(nn.Module):
    """Critic network for the supervised agent"""
    def __init__(self, state_size): 
        # Input: state; Output: state value (used to compute advantage)
        super(TwoHeadCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state):
        return self.net(state)


class TwoHeadMemory:
    def __init__(self): 
        self.states = []  # States at time t
        self.relocation_actions = []  # Sampled relocation actions at time t
        self.cf_actions = []  # Commission fee actions at time t
        self.rewards = []  # Rewards received at t+1 for actions taken at t
        self.next_states = []  # States at time t+1
        self.is_dones = []  # Whether episode ends at t+1
        self.relocation_logprobs = []  # Log-probabilities of relocation actions at t (used for advantage computation)
        self.cf_logprobs = []  # Log-probabilities of commission fee actions at t (used for advantage computation)
        self.state_values = []  # State values at time t
        self.next_state_values = []  # State values at time t+1
        
    def clear_memory(self): 
        del self.states[:]
        del self.relocation_actions[:]
        del self.cf_actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.is_dones[:]
        del self.relocation_logprobs[:]
        del self.cf_logprobs[:]
        del self.state_values[:]
        del self.next_state_values[:]
    
    def push_t(self, state, relocation_action, cf_action, relocation_logprob, cf_logprob, state_value): 
        # Store data available at time t
        self.states.append(state)  # Current state
        self.relocation_actions.append(relocation_action)  # Chosen relocation action
        self.cf_actions.append(cf_action)  # Chosen commission fee
        self.relocation_logprobs.append(relocation_logprob)  # Log-probability of relocation action
        self.cf_logprobs.append(cf_logprob)  # Log-probability of commission fee action
        self.state_values.append(state_value)  # Value of current state

    def push_tt(self, reward, next_state, done, next_state_value): 
        # Store data available at time t+1
        self.rewards.append(reward)  # Reward after taking action
        self.next_states.append(next_state)  # Next state
        self.is_dones.append(done)  # Whether episode terminates
        self.next_state_values.append(next_state_value)  # Value of next state

    def sample(self): 
        # Sample all stored experiences from PPO agent's memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.stack(self.states).to(device),
            torch.tensor(self.relocation_actions).to(device),
            torch.tensor(self.cf_actions).to(device),
            torch.tensor(self.rewards, dtype=torch.float32).to(device),
            torch.stack(self.next_states).to(device),
            torch.tensor(self.is_dones, dtype=torch.float32).to(device),
            torch.tensor(self.relocation_logprobs).to(device),
            torch.tensor(self.cf_logprobs).to(device),
            torch.tensor(self.state_values).to(device)
        )
