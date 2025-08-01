import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
class ActorNetwork(nn.Module):
    """Actor network with 3 FC layers + ReLU + Tanh output"""
    def __init__(self, state_dim=8, action_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.tanh(self.output(x))
    
class CriticNetwork(nn.Module):
    """Critic network with 3 FC layers + ReLU + linear output"""
    def __init__(self, state_dim=8, action_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)  # Q-value
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)
# Create file: training/lse_training/approach/ddpg_her.py
import torch
import numpy as np
from collections import deque
import random

class DDPG_HER_Trainer:
    def __init__(self, env, actor, critic, actor_target, critic_target):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        
        # Hyperparameters from paper
        self.buffer_size = 1000000
        self.batch_size = 256
        self.her_ratio = 0.8  # 80% HER samples
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
    def store_experience(self, st, agt, sgt, st_plus_1, action, reward):
        """Store (st, agt, sgt, st+1, action, reward) tuple"""
        self.replay_buffer.append((st, agt, sgt, st_plus_1, action, reward))
        
    def her_sample(self, episode_transitions):
        """Apply HER to episode transitions"""
        her_samples = []
        
        for i, (st, agt, sgt, st_plus_1, action, reward) in enumerate(episode_transitions):
            # Original sample
            her_samples.append((st, agt, sgt, st_plus_1, action, reward))
            
            # HER samples: replace goal with achieved goal
            if random.random() < self.her_ratio:
                # Pick a future transition as new goal
                future_idx = random.randint(i, len(episode_transitions)-1)
                future_agt = episode_transitions[future_idx][1]  # achieved goal
                
                # Recompute reward with new goal
                new_distance = np.linalg.norm(agt - future_agt)
                new_reward = -new_distance
                
                her_samples.append((st, agt, future_agt, st_plus_1, action, new_reward))
                
        return her_samples
    