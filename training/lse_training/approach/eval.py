"""
Evaluation script for Approach LSE
Evaluates the trained approach agent using hand-engineered manipulate/retract actions
"""
import torch
import numpy as np
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env import ApproachEnv
from training.training_env import CubeManipulationEnv  # For hand-engineered actions


def hand_engineered_manipulate(env, obs):
    """Pre-configured manipulate actions"""
    # Get current gripper and cube positions
    gripper_pos = obs.get('gripper_position', [0, 0, 0])
    cube_pos = obs['cube_position'][:3]
    target_pos = obs['target_position'][:3]
    
    # Simple hand-engineered manipulate: close gripper and move to target
    actions = []
    
    # Step 1: Close gripper
    action = [gripper_pos[0], gripper_pos[1], gripper_pos[2], 0, np.pi/2, np.pi/2, 0.0]  # Close gripper
    actions.append(action)
    
    # Step 2: Move to target position
    action = [target_pos[0], target_pos[1], target_pos[2] + 0.05, 0, np.pi/2, np.pi/2, 0.0]  # Move above target
    actions.append(action)
    
    # Step 3: Place object
    action = [target_pos[0], target_pos[1], target_pos[2] + 0.02, 0, np.pi/2, np.pi/2, 0.085]  # Place and open gripper
    actions.append(action)
    
    return actions


def hand_engineered_retract(env, obs):
    """Pre-configured retract actions"""
    # Get current gripper position
    gripper_pos = obs.get('gripper_position', [0, 0, 0])
    
    # Simple hand-engineered retract: move up and away
    actions = []
    
    # Step 1: Move up
    action = [gripper_pos[0], gripper_pos[1], gripper_pos[2] + 0.2, 0, np.pi/2, np.pi/2, 0.085]  # Move up
    actions.append(action)
    
    # Step 2: Move away
    action = [0, 0.3, 0.5, 0, np.pi/2, np.pi/2, 0.085]  # Move to safe position
    actions.append(action)
    
    return actions


def evaluate_approach_lse(trained_actor, num_episodes=10, log_dir=None):
    """Evaluate approach LSE with hand-engineered manipulate/retract"""
    
    # Initialize TensorBoard writer if log_dir provided
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir)
    
    # Initialize environment
    env = ApproachEnv(render_mode=None)  # No GUI for evaluation
    
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    approach_success_count = 0
    
    print(f"🎯 Starting evaluation of {num_episodes} episodes...")
    
    # Progress bar for episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating episodes"):
        episode_reward = 0
        episode_length = 0
        
        # Reset environment
        obs, _ = env.reset()
        
        # Phase 1: Approach (trained agent)
        approach_complete = False
        approach_steps = 0
        
        for step in range(100):  # Approach phase
            # Get action from trained actor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = trained_actor(state_tensor).numpy()[0]
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            approach_steps += 1
            
            if info.get('approach_complete', False):
                approach_complete = True
                approach_success_count += 1
                break
        
        if not approach_complete:
            if writer:
                writer.add_scalar('Evaluation/Approach_Failed', 1, episode)
            continue
        
        # Phase 2: Manipulate (hand-engineered)
        manipulate_actions = hand_engineered_manipulate(env, obs)
        manipulate_complete = False
        
        for action in manipulate_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if info.get('manipulate_complete', False) or info.get('task_completed', False):
                manipulate_complete = True
                break
        
        if not manipulate_complete:
            if writer:
                writer.add_scalar('Evaluation/Manipulate_Failed', 1, episode)
            continue
        
        # Phase 3: Retract (hand-engineered)
        retract_actions = hand_engineered_retract(env, obs)
        retract_complete = False
        
        for action in retract_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if info.get('retract_complete', False) or info.get('task_completed', False):
                retract_complete = True
                break
        
        # Check if task was successful
        if info.get('task_completed', False):
            success_count += 1
            if writer:
                writer.add_scalar('Evaluation/Episode_Success', 1, episode)
        else:
            if writer:
                writer.add_scalar('Evaluation/Episode_Success', 0, episode)
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if writer:
            writer.add_scalar('Evaluation/Episode_Reward', episode_reward, episode)
            writer.add_scalar('Evaluation/Episode_Length', episode_length, episode)
            writer.add_scalar('Evaluation/Approach_Steps', approach_steps, episode)
    
    # Calculate success rates
    success_rate = success_count / num_episodes
    approach_success_rate = approach_success_count / num_episodes
    
    # Log final metrics
    if writer:
        writer.add_scalar('Evaluation/Final_Success_Rate', success_rate, 0)
        writer.add_scalar('Evaluation/Approach_Success_Rate', approach_success_rate, 0)
        writer.add_scalar('Evaluation/Average_Episode_Reward', np.mean(episode_rewards), 0)
        writer.add_scalar('Evaluation/Average_Episode_Length', np.mean(episode_lengths), 0)
        writer.close()
    
    # Print results
    print(f"\n🎯 Approach LSE Evaluation Results:")
    print(f"   Overall Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    print(f"   Approach Success Rate: {approach_success_rate:.2f} ({approach_success_count}/{num_episodes})")
    print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    
    # Determine if approach needs more training
    if success_rate < 0.8:
        print("❌ Approach LSE needs more training!")
        return False
    else:
        print("✅ Approach LSE training complete!")
        return True


def evaluate_approach_only(trained_actor, num_episodes=10, log_dir=None):
    """Evaluate only the approach phase (without manipulate/retract)"""
    
    # Initialize TensorBoard writer if log_dir provided
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir)
    
    env = ApproachEnv(render_mode=None)
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    
    print(f"🎯 Starting approach-only evaluation of {num_episodes} episodes...")
    
    # Progress bar for episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating approach"):
        episode_reward = 0
        episode_length = 0
        
        obs, _ = env.reset()
        
        for step in range(100):
            # Get action from trained actor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = trained_actor(state_tensor).numpy()[0]
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if info.get('approach_complete', False):
                success_count += 1
                if writer:
                    writer.add_scalar('Evaluation/Approach_Success', 1, episode)
                break
        
        if step == 99:  # Max steps reached
            if writer:
                writer.add_scalar('Evaluation/Approach_Success', 0, episode)
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if writer:
            writer.add_scalar('Evaluation/Approach_Episode_Reward', episode_reward, episode)
            writer.add_scalar('Evaluation/Approach_Episode_Length', episode_length, episode)
    
    success_rate = success_count / num_episodes
    
    # Log final metrics
    if writer:
        writer.add_scalar('Evaluation/Approach_Final_Success_Rate', success_rate, 0)
        writer.add_scalar('Evaluation/Approach_Average_Reward', np.mean(episode_rewards), 0)
        writer.add_scalar('Evaluation/Approach_Average_Length', np.mean(episode_lengths), 0)
        writer.close()
    
    print(f"\n🎯 Approach-Only Evaluation Results:")
    print(f"   Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    
    return success_rate


if __name__ == "__main__":
    # Example usage
    print("🚀 Approach LSE Evaluation Script")
    print("This script evaluates the trained approach agent")
    print("Make sure you have a trained actor model before running this!")
    
    # You would load your trained actor here
    # trained_actor = load_trained_actor("path/to/model.pth")
    
    # Then run evaluation with TensorBoard logging
    # log_dir = "logs/evaluation_" + time.strftime("%Y%m%d_%H%M%S")
    # evaluate_approach_lse(trained_actor, log_dir=log_dir)
    # evaluate_approach_only(trained_actor, log_dir=log_dir)