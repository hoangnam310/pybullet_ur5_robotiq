#!/usr/bin/env python3
"""
Continuous Best Model Monitor

This script continuously tests and visualizes your best model while training runs.
Run this in a separate terminal to watch your agent improve in real-time!
"""

import time
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO
    from env_wrapper import CubeManipulationEnv, CubeManipulationEnvSimple
except ImportError:
    print("❌ Please install required packages: pip install stable-baselines3 gymnasium")
    sys.exit(1)

def find_best_model():
    """Find the most recent best model"""
    models_dir = "training/models"
    if not os.path.exists(models_dir):
        return None
    
    # Look for best model directories
    best_dirs = [d for d in os.listdir(models_dir) if d.startswith("best_") and os.path.isdir(os.path.join(models_dir, d))]
    
    if not best_dirs:
        return None
    
    # Find the most recent one
    best_dirs.sort(reverse=True)  # Sort by timestamp (newest first)
    
    for best_dir in best_dirs:
        model_path = os.path.join(models_dir, best_dir, "best_model.zip")
        if os.path.exists(model_path):
            return model_path
    
    return None

def test_model(model_path, episodes=3):
    """Test the model and return performance stats"""
    try:
        # Load model
        model = PPO.load(model_path)
        
        # Create environment with GUI
        try:
            env = CubeManipulationEnvSimple(render_mode='human')
            env_type = "Simple"
        except:
            env = CubeManipulationEnv(render_mode='human')
            env_type = "Full"
        
        print(f"🎮 Testing {env_type} environment with {episodes} episodes...")
        
        # Run episodes
        total_reward = 0
        successes = 0
        total_steps = 0
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_steps += 1
                
                # Small delay for visualization
                time.sleep(0.01)
            
            total_reward += episode_reward
            total_steps += episode_steps
            
            if info.get('is_success', False):
                successes += 1
            
            print(f"   Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}, Success={'✅' if info.get('is_success', False) else '❌'}")
        
        # Calculate stats
        avg_reward = total_reward / episodes
        success_rate = (successes / episodes) * 100
        avg_steps = total_steps / episodes
        
        env.close()
        
        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'episodes': episodes
        }
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return None

def main():
    """Main monitoring loop"""
    print("🔍 Best Model Monitor Started")
    print("Watching for best model updates during training...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    last_modified = 0
    test_count = 0
    
    try:
        while True:
            # Find best model
            model_path = find_best_model()
            
            if not model_path:
                print("⏳ Waiting for best model to be created...")
                time.sleep(10)
                continue
            
            # Check if model was updated
            current_modified = os.path.getmtime(model_path)
            
            if current_modified > last_modified:
                test_count += 1
                print(f"\n🔄 Test #{test_count} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"📂 Found updated model: {os.path.basename(os.path.dirname(model_path))}")
                
                # Test the model
                stats = test_model(model_path, episodes=3)
                
                if stats:
                    print(f"📊 Performance Summary:")
                    print(f"   Average Reward: {stats['avg_reward']:.2f}")
                    print(f"   Success Rate: {stats['success_rate']:.1f}%")
                    print(f"   Average Steps: {stats['avg_steps']:.1f}")
                    
                    # Performance indicators
                    if stats['success_rate'] > 0:
                        print("🎉 Robot is starting to succeed!")
                    elif stats['avg_reward'] > 10:
                        print("📈 Robot is getting better rewards!")
                    elif stats['avg_reward'] > 5:
                        print("🔄 Robot is learning...")
                    else:
                        print("🤖 Robot is still exploring...")
                
                last_modified = current_modified
            
            # Wait before checking again
            print(f"⏳ Waiting 30 seconds for next check... (Ctrl+C to stop)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main() 