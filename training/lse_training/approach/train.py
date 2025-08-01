"""
Training script for Approach LSE using DDPG + HER
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from current directory
from env import ApproachEnv


class ActorNetwork(nn.Module):
    """Actor network for approach LSE - 3 FC layers + ReLU + Tanh output"""
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
        return torch.tanh(self.output(x))  # Tanh activation for actions


class CriticNetwork(nn.Module):
    """Critic network for approach LSE - 3 FC layers + ReLU + linear output"""
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
        return self.output(x)  # No activation for critic


class DDPG_HER_Agent:
    """DDPG + HER agent for approach LSE training"""
    
    def __init__(self, env, log_dir=None):
        self.env = env
        self.state_dim = 11  # From approach environment: [gripper_pos(3), cube_pos(3), distance(1), height_diff(1), wrist_1, wrist_2, wrist_3(3)]
        self.action_dim = 6  # [x, y, z, roll, pitch, yaw]
        
        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim, self.action_dim)
        self.actor_target = ActorNetwork(self.state_dim, self.action_dim)
        self.critic_target = CriticNetwork(self.state_dim, self.action_dim)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)  # Higher learning rate
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Hyperparameters
        self.buffer_size = 1000000
        self.batch_size = 256  # Larger batch size for more stable updates
        self.her_ratio = 0.8  # 80% HER samples
        self.gamma = 0.98  # Higher discount factor for better long-term planning
        self.tau = 0.001  # Slower target network updates for stability
        self.noise_std = 0.8  # Moderate exploration noise
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # TensorBoard logging
        self.writer = None
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        
        # Training stats
        self.total_steps = 0
        self.episode_count = 0
        
        print("🤖 DDPG + HER Agent initialized!")
        print(f"   State dimension: {self.state_dim}")
        print(f"   Action dimension: {self.action_dim}")
        print(f"   Buffer size: {self.buffer_size}")
        print(f"   HER ratio: {self.her_ratio}")
    
    def store_experience(self, st, agt, sgt, st_plus_1, action, reward, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((st, agt, sgt, st_plus_1, action, reward, done))
    
    def her_sample(self, episode_transitions):
        """Apply HER to episode transitions"""
        her_samples = []
        
        for i, (st, agt, sgt, st_plus_1, action, reward, done) in enumerate(episode_transitions):
            # Original sample
            her_samples.append((st, agt, sgt, st_plus_1, action, reward, done))
            
            # HER samples: replace goal with achieved goal
            if random.random() < self.her_ratio and not done:
                # Pick a future transition as new goal
                future_idx = random.randint(i, len(episode_transitions)-1)
                future_agt = episode_transitions[future_idx][1]  # achieved goal (gripper position)
                
                # Create new state with replaced goal (cube position)
                # st format: [gripper_pos(3), cube_pos(3), distance(1), height_diff(1), wrist_angles(3)]
                new_st = st.copy()
                new_st[3:6] = future_agt  # Replace cube position with achieved goal
                
                # Recompute distance and height difference
                gripper_pos = st[0:3]  # Current gripper position
                new_distance = np.linalg.norm(gripper_pos - future_agt)
                new_height_diff = gripper_pos[2] - future_agt[2]
                
                # Update distance and height difference in observation
                new_st[6] = new_distance
                new_st[7] = new_height_diff
                
                # Recompute reward with new goal
                new_reward = -new_distance
                
                # Add success bonus if close enough
                if new_distance < 0.05:  # approach_distance_threshold
                    new_reward += 10
                
                her_samples.append((new_st, agt, future_agt, st_plus_1, action, new_reward, done))
                
        return her_samples
    
    def update_policy(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        st_batch = torch.FloatTensor([exp[0] for exp in batch])
        agt_batch = torch.FloatTensor([exp[1] for exp in batch])
        sgt_batch = torch.FloatTensor([exp[2] for exp in batch])
        st_plus_1_batch = torch.FloatTensor([exp[3] for exp in batch])
        action_batch = torch.FloatTensor([exp[4] for exp in batch])
        reward_batch = torch.FloatTensor([exp[5] for exp in batch])
        done_batch = torch.FloatTensor([exp[6] for exp in batch])
        
        # Update critic
        next_actions = self.actor_target(st_plus_1_batch)
        next_q_values = self.critic_target(st_plus_1_batch, next_actions)
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values.squeeze()
        current_q_values = self.critic(st_batch, action_batch).squeeze()
        
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(st_batch)
        actor_loss = -self.critic(st_batch, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log losses
        if self.writer:
            self.writer.add_scalar('Training/Critic_Loss', critic_loss.item(), self.total_steps)
            self.writer.add_scalar('Training/Actor_Loss', actor_loss.item(), self.total_steps)
    
    def train(self, total_steps=15000, eval_interval=15000):
        """Train the agent"""
        print(f"🚀 Starting training for {total_steps} steps...")
        
        # Calculate epochs for progress tracking
        steps_per_epoch = eval_interval
        total_epochs = total_steps // steps_per_epoch
        current_epoch = 0
        
        print(f"📊 Training for {total_epochs} epochs ({total_steps:,} total steps)")
        
        # Progress bar for total training
        with tqdm(total=total_steps, desc="Training progress") as pbar:
            while self.total_steps < total_steps:
                # Reset environment
                obs, _ = self.env.reset()
                episode_transitions = []
                episode_reward = 0
                episode_length = 0
                
                # Episode loop (300 steps max as per paper)
                for step in range(300):
                    # Get action from actor with exploration noise
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = self.actor(state_tensor).detach().numpy()[0]
                    
                    # Improved exploration strategy
                    # Start with high exploration, gradually reduce
                    epsilon = max(0.05, 1.0 - (self.total_steps / total_steps) * 0.8)  # Start at 100%, decay to 5%
                    
                    if np.random.random() < epsilon:
                        # Random actions for exploration
                        action = np.random.uniform(-1, 1, size=action.shape)
                    else:
                        # Add exploration noise with adaptive decay
                        # Keep more exploration early, reduce later
                        progress = self.total_steps / total_steps
                        decay_factor = max(0.1, 1.0 - progress * 0.8)  # Keep 10% minimum noise
                        current_noise_std = self.noise_std * decay_factor
                        noise = np.random.normal(0, current_noise_std, size=action.shape)
                        action = np.clip(action + noise, -1, 1)
                    
                    # Take step in environment
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Extract components for HER
                    st = obs  # current state (11-dimensional approach observation)
                    agt = info.get('gripper_position', [0, 0, 0])  # achieved goal (gripper position)
                    sgt = obs[3:6]  # target goal (cube position in approach obs format)
                    st_plus_1 = next_obs  # next state (11-dimensional approach observation)
                    
                    # Store transition
                    episode_transitions.append((st, agt, sgt, st_plus_1, action, reward, done))
                    
                    episode_reward += reward
                    episode_length += 1
                    self.total_steps += 1
                    
                    # Update environment's training step for progressive penalties
                    self.env.set_training_step(self.total_steps)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                # Apply HER and store in replay buffer
                her_samples = self.her_sample(episode_transitions)
                for sample in her_samples:
                    self.store_experience(*sample)
                
                # Update policy every 100 steps (more frequent updates)
                if self.total_steps % 100 == 0:
                    self.update_policy()
                
                # Log episode metrics
                if self.writer:
                    self.writer.add_scalar('Training/Episode_Reward', episode_reward, self.episode_count)
                    self.writer.add_scalar('Training/Episode_Length', episode_length, self.episode_count)
                    self.writer.add_scalar('Training/Replay_Buffer_Size', len(self.replay_buffer), self.episode_count)
                    self.writer.add_scalar('Training/Distance_To_Cube', info.get('distance_to_cube', 0), self.episode_count)
                    self.writer.add_scalar('Training/Too_Far_Away', 1.0 if info.get('too_far_away', False) else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Floor_Collision', 1.0 if info.get('floor_collision', False) else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Self_Collision', 1.0 if info.get('self_collision', False) else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Body_Collision', 1.0 if info.get('body_collision', False) else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Any_Collision', 1.0 if info.get('any_collision', False) else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Proper_Pose', 1.0 if info.get('proper_pose', False) else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Gripper_Height', info.get('gripper_position', [0, 0, 0])[2], self.episode_count)
                    
                    # Enhanced logging for orientation and positioning
                    gripper_pos = info.get('gripper_position', [0, 0, 0])
                    cube_pos = obs[3:6]  # Cube position from observation (approach format)
                    height_diff = gripper_pos[2] - cube_pos[2]
                    horizontal_distance = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])
                    
                    self.writer.add_scalar('Training/Height_Difference', height_diff, self.episode_count)
                    self.writer.add_scalar('Training/Horizontal_Distance', horizontal_distance, self.episode_count)
                    self.writer.add_scalar('Training/Above_Cube', 1.0 if height_diff > 0.02 else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Close_Horizontally', 1.0 if horizontal_distance < 0.05 else 0.0, self.episode_count)
                    self.writer.add_scalar('Training/Sweet_Spot', 1.0 if (0.03 < height_diff < 0.07 and horizontal_distance < 0.03) else 0.0, self.episode_count)
                
                self.episode_count += 1
                
                # Evaluation every eval_interval steps (at the end of each epoch)
                if self.total_steps % eval_interval == 0:
                    current_epoch = self.total_steps // eval_interval
                    print(f"\n📊 Epoch {current_epoch}/{total_epochs} completed at step {self.total_steps:,}")
                    self.evaluate()
                    
                    # Auto-save model every epoch
                    checkpoint_path = os.path.join(self.writer.log_dir, f"checkpoint_epoch_{current_epoch}.pth")
                    self.save_model(checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                    
                    # Check if we've completed all epochs
                    if current_epoch >= total_epochs:
                        break
        
        print(f"\n✅ Training completed! Total steps: {self.total_steps:,}")
        print(f"📊 Completed {current_epoch} epochs out of {total_epochs} planned")
        if self.writer:
            self.writer.close()
    
    def evaluate(self, num_episodes=5):
        """Quick evaluation during training"""
        success_count = 0
        too_far_count = 0
        collision_count = 0
        timeout_count = 0
        total_reward = 0
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(100):
                # Get action without noise
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor(state_tensor).numpy()[0]
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if terminated:
                    termination_reason = info.get('termination_reason', 'unknown')
                    if termination_reason == 'success':
                        success_count += 1
                    elif termination_reason == 'too_far':
                        too_far_count += 1
                    elif termination_reason == 'collision':
                        collision_count += 1
                    elif termination_reason == 'timeout':
                        timeout_count += 1
                    break
            
            total_reward += episode_reward
        
        success_rate = success_count / num_episodes
        too_far_rate = too_far_count / num_episodes
        collision_rate = collision_count / num_episodes
        timeout_rate = timeout_count / num_episodes
        avg_reward = total_reward / num_episodes
        
        if self.writer:
            self.writer.add_scalar('Evaluation/Success_Rate', success_rate, self.total_steps)
            self.writer.add_scalar('Evaluation/Too_Far_Rate', too_far_rate, self.total_steps)
            self.writer.add_scalar('Evaluation/Collision_Rate', collision_rate, self.total_steps)
            self.writer.add_scalar('Evaluation/Timeout_Rate', timeout_rate, self.total_steps)
            self.writer.add_scalar('Evaluation/Average_Reward', avg_reward, self.total_steps)
        
        print(f"   Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
        print(f"   Too Far Rate: {too_far_rate:.2f} ({too_far_count}/{num_episodes})")
        print(f"   Collision Rate: {collision_rate:.2f} ({collision_count}/{num_episodes})")
        print(f"   Timeout Rate: {timeout_rate:.2f} ({timeout_count}/{num_episodes})")
        print(f"   Average Reward: {avg_reward:.2f}")
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        print(f"📂 Model loaded from {path}")


def main():
    """Main training function"""
    print("🚀 Approach LSE Training Script")
    
    # Create log directory
    log_dir = f"logs/approach_training_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    epochs = 10  # Reduced from 50 to 20 epochs for faster testing
    steps_per_epoch = 15000  # 15k steps per epoch as per paper
    total_steps = epochs * steps_per_epoch  # Total steps for all epochs
    eval_interval = steps_per_epoch  # Evaluate every epoch
    # Initialize environment and agent
    env = ApproachEnv(render_mode=None)  # No GUI for training
    env.set_total_training_steps(total_steps)
    
    agent = DDPG_HER_Agent(env, log_dir=log_dir)
    
    # Training parameters
    
    
    print(f"📁 Logs will be saved to: {log_dir}")
    print(f"🎯 Training for {epochs} epochs ({total_steps:,} total steps)")
    print(f"📊 Evaluation every {eval_interval:,} steps (every epoch)")
    print(f"📈 Expected completion: {epochs} epochs = {total_steps:,} steps")
    
    # Start training
    agent.train(total_steps=total_steps, eval_interval=eval_interval)
    
    # Save final model
    model_path = os.path.join(log_dir, "final_model.pth")
    agent.save_model(model_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Logs: {log_dir}")
    print(f"💾 Test Model: python3 test_gui.py --model {model_path} --interactive")
    print(f"📈 View training progress: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main() 