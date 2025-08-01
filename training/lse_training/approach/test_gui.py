"""
GUI Test Script for Approach LSE
Visualize the trained approach agent in real-time with PyBullet GUI
python test_gui.py --model logs/approach_training_20241201_143022/final_model.pth --interactive
"""
import torch
import numpy as np
import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env import ApproachEnv
from train import ActorNetwork, DDPG_HER_Agent


def load_trained_agent(model_path):
    """Load a trained agent from saved model"""
    # Initialize environment and agent
    env = ApproachEnv(render_mode='human')  # Enable GUI
    agent = DDPG_HER_Agent(env)
    
    # Load trained model
    agent.load_model(model_path)
    
    return agent


def print_environment_info(env):
    """Print current environment configuration and state"""
    print("🔧 Environment Configuration:")
    print(f"   Position scale: {env.position_scale}m per step")
    print(f"   Orientation scale: {env.orientation_scale}rad per step")
    print(f"   Approach threshold: {env.approach_distance_threshold}m")
    print(f"   Too far threshold: {env.too_far_threshold}m")
    print(f"   Min gripper height: {env.min_gripper_height}m")
    print(f"   Collision penalty: {env.collision_penalty}")
    print(f"   Self-collision penalty: {env.self_collision_penalty}")
    print(f"   Orientation reward scale: {env.orientation_reward_scale}")
    print(f"   Enable orientation reward: {env.enable_orientation_reward}")
    print(f"   Max episode length: {env.max_episode_length} steps")


def test_agent_gui(agent, num_episodes=5, delay=0.1, verbose=True, debug_collision=True):
    """Test trained agent with GUI visualization"""
    print("🎮 Starting GUI test of trained approach agent...")
    print(f"   Episodes: {num_episodes}")
    print(f"   Step delay: {delay}s")
    print("   Press Ctrl+C to stop")
    
    success_count = 0
    termination_reasons = {
        'success': 0,
        'collision': 0,
        'too_far': 0,
        'timeout': 0,
        'unknown': 0
    }
    
    for episode in range(num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, _ = agent.env.reset()
        episode_reward = 0
        step_count = 0
        
        # Episode loop
        for step in range(300):  # Max 300 steps as per paper
            # Get action from trained actor (no noise)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = agent.actor(state_tensor).numpy()[0]
            
            # Take step in environment
            obs, reward, terminated, truncated, info = agent.env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Print step info
            if verbose:
                distance = info.get('distance_to_cube', 0)
                gripper_pos = info.get('gripper_position', [0, 0, 0])
                print(f"   Step {step + 1}: Action={action}, Reward={reward:.3f}, Distance={distance:.3f}, Gripper={gripper_pos}")
            
            # Check if approach is complete
            if info.get('approach_complete', False):
                success_count += 1
                termination_reasons['success'] += 1
                print(f"   ✅ Approach completed in {step + 1} steps!")
                break
            
            # Add delay for visualization
            time.sleep(delay)
            
            if terminated or truncated:
                # Get termination reason and details
                termination_reason = info.get('termination_reason', 'unknown')
                distance_to_cube = info.get('distance_to_cube', 0)
                gripper_height = info.get('gripper_position', [0, 0, 0])[2]
                floor_collision = info.get('floor_collision', False)
                self_collision = info.get('self_collision', False)
                body_collision = info.get('body_collision', False)
                too_far_away = info.get('too_far_away', False)
                
                # Debug collision detection if requested
                if debug_collision and (floor_collision or self_collision or body_collision):
                    agent.env.debug_collision_detection()
                
                # Build collision info string
                collision_info = []
                if floor_collision:
                    collision_info.append("FLOOR")
                if self_collision:
                    collision_info.append("SELF")
                collision_str = f"[{', '.join(collision_info)}]" if collision_info else ""
                
                print(f"   ❌ Episode terminated after {step + 1} steps")
                print(f"      Reason: {termination_reason.upper()} {collision_str}")
                print(f"      Distance to cube: {distance_to_cube:.3f}m")
                print(f"      Gripper height: {gripper_height:.3f}m")
                
                # Track termination reason
                if termination_reason in termination_reasons:
                    termination_reasons[termination_reason] += 1
                else:
                    termination_reasons['unknown'] += 1
                break
        
        print(f"   Episode {episode + 1} finished: Reward={episode_reward:.3f}, Steps={step_count}")
        
        # Wait between episodes
        if episode < num_episodes - 1:
            print("   Waiting 3 seconds before next episode...")
            time.sleep(3)
    
    # Final results
    success_rate = success_count / num_episodes
    print(f"\n🎯 GUI Test Results:")
    print(f"   Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    print(f"   Termination Reasons:")
    for reason, count in termination_reasons.items():
        if count > 0:
            print(f"     {reason.upper()}: {count} episodes")
    
    return success_rate


def test_agent_interactive(agent, verbose=True):
    """Interactive test where you can control the agent manually"""
    print("🎮 Interactive GUI Test")
    print("   The agent will run automatically, but you can observe the behavior")
    print("   Press Ctrl+C to stop")
    
    try:
        while True:
            # Reset environment
            obs, _ = agent.env.reset()
            print("🔄 Environment reset - starting new episode")
            
            # Run one episode
            for step in range(300):
                # Get action from trained actor
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action = agent.actor(state_tensor).numpy()[0]
                
                # Take step
                obs, reward, terminated, truncated, info = agent.env.step(action)
                
                # Print info every 10 steps or if verbose
                if verbose or step % 10 == 0:
                    distance = info.get('distance_to_cube', 0)
                    gripper_pos = info.get('gripper_position', [0, 0, 0])
                    print(f"   Step {step}: Distance={distance:.3f}, Reward={reward:.3f}, Gripper={gripper_pos}")
                
                if info.get('approach_complete', False):
                    print(f"   ✅ Approach completed in {step + 1} steps!")
                    break
                
                if terminated or truncated:
                    # Get termination reason and details
                    termination_reason = info.get('termination_reason', 'unknown')
                    distance_to_cube = info.get('distance_to_cube', 0)
                    gripper_height = info.get('gripper_position', [0, 0, 0])[2]
                    floor_collision = info.get('floor_collision', False)
                    self_collision = info.get('self_collision', False)
                    body_collision = info.get('body_collision', False)
                    too_far_away = info.get('too_far_away', False)
                    
                    # Build collision info string
                    collision_info = []
                    if floor_collision:
                        collision_info.append("FLOOR")
                    if self_collision:
                        collision_info.append("SELF")
                    if body_collision:
                        collision_info.append("BODY")
                    collision_str = f"[{', '.join(collision_info)}]" if collision_info else ""
                    
                    print(f"   ❌ Episode terminated after {step + 1} steps")
                    print(f"      Reason: {termination_reason.upper()} {collision_str}")
                    print(f"      Distance to cube: {distance_to_cube:.3f}m")
                    print(f"      Gripper height: {gripper_height:.3f}m")
                    break
                
                time.sleep(0.05)  # Fast visualization
            
            print("   Episode finished. Press Ctrl+C to stop or wait 2 seconds for next episode...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Interactive test stopped by user")


def main():
    """Main function for GUI testing"""
    parser = argparse.ArgumentParser(description="Test trained approach agent with GUI")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (seconds)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--verbose", action="store_true", help="Show detailed step-by-step information")
    parser.add_argument("--debug-collision", action="store_true", help="Debug collision detection")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("   Please train a model first using: python train.py")
        return
    
    try:
        # Load trained agent
        print(f"📂 Loading model from: {args.model}")
        agent = load_trained_agent(args.model)
        
        # Print environment configuration
        print_environment_info(agent.env)
        print()
        
        if args.interactive:
            # Interactive mode
            test_agent_interactive(agent, args.verbose)
        else:
            # Standard test mode
            test_agent_gui(agent, args.episodes, args.delay, args.verbose, args.debug_collision)
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("   Make sure you have a properly trained model")


if __name__ == "__main__":
    main() 