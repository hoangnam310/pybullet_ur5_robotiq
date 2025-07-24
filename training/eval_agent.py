#!/usr/bin/env python3
"""
Cube Manipulation Agent Evaluation Script

This script tests your trained agent and shows how well it learned
to manipulate the cube. It's like the final exam for your AI!

Usage:
    python eval_agent.py                         # Test latest model
    python eval_agent.py --model best            # Test best model
    python eval_agent.py --model MODEL_PATH      # Test specific model
    python eval_agent.py --episodes 10           # Test with 10 episodes
    python eval_agent.py --render                # Show visualization
    python eval_agent.py --model "training/models/best_cube_manipulation_agent_short_20250724_122113/best_model.zip" --render --episodes 5
"""

import argparse
import os
import sys
import time
import numpy as np
from datetime import datetime
import glob

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import RL libraries
try:
    from stable_baselines3 import PPO
    print("✅ Stable-Baselines3 imported successfully!")
except ImportError:
    print("❌ Error: stable-baselines3 not installed!")
    print("Please install it with: pip install stable-baselines3[extra]")
    sys.exit(1)

# Import our modules
from config import TrainingConfig, get_model_path, detect_device
from env_wrapper import CubeManipulationEnv, CubeManipulationEnvSimple

class CubeEvaluator:
    """Evaluation class for testing trained agents"""
    
    def __init__(self, model_path, render=False, episodes=10):
        self.model_path = model_path
        self.render = render
        self.episodes = episodes
        
        # Detect device for model loading
        self.device = detect_device()
        
        print(f"🔬 Cube Manipulation Agent Evaluator")
        print(f"📂 Model: {model_path}")
        print(f"💾 Device: {self.device}")
        print(f"📺 Render: {render}")
        print(f"🎯 Episodes: {episodes}")
        print("=" * 50)
    
    def load_model_and_env(self):
        """Load the trained model and create evaluation environment"""
        print("📂 Loading model...")
        
        try:
            # Load the model with device support
            model = PPO.load(self.model_path, device=self.device)
            print(f"✅ Model loaded successfully on {self.device}!")
            
            # Determine environment type from model
            # For simplicity, we'll try both and see which works
            render_mode = 'human' if self.render else None
            
            if self.render:
                print("📺 Setting up GUI visualization...")
                if render_mode == 'human':
                    print("🖥️  GUI window will open - use it to watch your agent!")
            
            try:
                # Try simple environment first
                env = CubeManipulationEnvSimple(render_mode=render_mode)
                # Test if model is compatible
                obs, _ = env.reset()
                action = model.predict(obs, deterministic=True)[0]
                print("🎯 Using Simple environment")
            except Exception as simple_error:
                try:
                    # Try full environment
                    env = CubeManipulationEnv(render_mode=render_mode)
                    obs, _ = env.reset()
                    action = model.predict(obs, deterministic=True)[0]
                    print("🤖 Using Full environment")
                except Exception as full_error:
                    print(f"❌ Failed to create environment: {full_error}")
                    return None, None
            
            return model, env
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Make sure the model file exists and is compatible")
            return None, None
    
    def evaluate_agent(self, model, env):
        """Run evaluation episodes and collect statistics"""
        print(f"\n🚀 Starting evaluation with {self.episodes} episodes...")
        
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_count': 0,
            'final_distances': [],
            'total_steps': 0
        }
        
        for episode in range(self.episodes):
            print(f"\n📊 Episode {episode + 1}/{self.episodes}")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            start_time = time.time()
            
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Print progress occasionally
                if episode_length % 100 == 0 and self.render:
                    distance = info.get('cube_distance', 'unknown')
                    print(f"   Step {episode_length}: Distance = {distance:.3f}")
                
                # Safety limit
                if episode_length > 2000:
                    print("   ⚠️  Episode too long, stopping")
                    break
            
            # Record results
            results['episode_rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)
            results['total_steps'] += episode_length
            
            # Check success
            is_success = info.get('is_success', False)
            if is_success:
                results['success_count'] += 1
                print(f"   ✅ SUCCESS! Reward: {episode_reward:.2f}, Steps: {episode_length}")
            else:
                final_distance = info.get('cube_distance', 'unknown')
                results['final_distances'].append(final_distance)
                print(f"   ❌ Failed. Reward: {episode_reward:.2f}, Steps: {episode_length}, Final distance: {final_distance:.3f}")
            
            episode_time = time.time() - start_time
            print(f"   ⏱️  Episode time: {episode_time:.1f}s")
        
        return results
    
    def print_statistics(self, results):
        """Print evaluation statistics"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        
        # Success rate
        success_rate = (results['success_count'] / self.episodes) * 100
        print(f"🎯 Success Rate: {success_rate:.1f}% ({results['success_count']}/{self.episodes})")
        
        # Reward statistics
        rewards = results['episode_rewards']
        print(f"🏆 Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"📈 Best Reward: {np.max(rewards):.2f}")
        print(f"📉 Worst Reward: {np.min(rewards):.2f}")
        
        # Episode length statistics
        lengths = results['episode_lengths']
        print(f"⏱️  Average Episode Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} steps")
        print(f"🏃 Shortest Episode: {np.min(lengths)} steps")
        print(f"🐌 Longest Episode: {np.max(lengths)} steps")
        
        # Distance statistics (for failed episodes)
        if results['final_distances']:
            distances = results['final_distances']
            print(f"📏 Average Final Distance (failures): {np.mean(distances):.3f} ± {np.std(distances):.3f}")
        
        # Overall performance assessment
        print("\n📝 PERFORMANCE ASSESSMENT:")
        if success_rate >= 80:
            print("🌟 EXCELLENT! Your agent is performing very well!")
        elif success_rate >= 60:
            print("👍 GOOD! Your agent is learning well, but could improve")
        elif success_rate >= 30:
            print("📚 LEARNING! Your agent shows promise but needs more training")
        elif success_rate >= 10:
            print("🔧 NEEDS WORK! Try training longer or adjusting hyperparameters")
        else:
            print("❌ POOR! Something might be wrong - check your training setup")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if success_rate < 50:
            print("• Try training for more timesteps")
            print("• Experiment with different reward shaping")
            print("• Check if the environment is working correctly")
        elif success_rate < 80:
            print("• Continue training to improve performance")
            print("• Try fine-tuning hyperparameters")
        else:
            print("• Great job! Try the full environment if using simple")
            print("• Experiment with more complex tasks")
    
    def run_evaluation(self):
        """Main evaluation function"""
        # Load model and environment
        model, env = self.load_model_and_env()
        if model is None or env is None:
            return False
        
        try:
            # Run evaluation
            results = self.evaluate_agent(model, env)
            
            # Print statistics
            self.print_statistics(results)
            
            # Clean up
            env.close()
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Evaluation interrupted by user")
            env.close()
            return False
        
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            env.close()
            return False

def find_model_path(model_name):
    """Find the actual path for a model"""
    config = TrainingConfig()
    
    if model_name == "best":
        # Find the best model
        pattern = os.path.join(config.MODEL_DIR, "best_*")
        best_models = glob.glob(pattern)
        if best_models:
            # Return the most recent best model
            return max(best_models, key=os.path.getctime)
        else:
            print("❌ No best model found")
            return None
    
    elif model_name == "latest":
        # Find the most recent model
        pattern = os.path.join(config.MODEL_DIR, "*")
        all_models = glob.glob(pattern)
        if all_models:
            return max(all_models, key=os.path.getctime)
        else:
            print("❌ No models found")
            return None
    
    elif os.path.exists(model_name):
        # Direct path provided
        return model_name
    
    else:
        # Try to find model in model directory
        model_path = os.path.join(config.MODEL_DIR, model_name)
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"❌ Model not found: {model_name}")
            return None

def list_available_models():
    """List all available trained models"""
    config = TrainingConfig()
    
    if not os.path.exists(config.MODEL_DIR):
        print("❌ No models directory found. Train a model first!")
        return
    
    models = glob.glob(os.path.join(config.MODEL_DIR, "*"))
    
    if not models:
        print("❌ No trained models found. Train a model first!")
        print(f"💡 Run: python train_agent.py --quick-test")
        return
    
    print(f"📂 Available models in {config.MODEL_DIR}:")
    for i, model_path in enumerate(sorted(models)):
        model_name = os.path.basename(model_path)
        size_mb = os.path.getsize(model_path) / (1024*1024) if os.path.isfile(model_path) else 0
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"  {i+1}. {model_name} ({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M')})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate cube manipulation agent")
    parser.add_argument('--model', type=str, default='best',
                       help='Model to evaluate (best/latest/path)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Show GUI visualization during evaluation')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU usage (disable MPS/CUDA)')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    # Handle device override
    if args.no_gpu:
        TrainingConfig.USE_MPS = False
        TrainingConfig.DEVICE = "cpu"
        print("🔧 GPU disabled, using CPU only")
    
    # List models if requested
    if args.list:
        list_available_models()
        return
    
    # Find model path
    model_path = find_model_path(args.model)
    if not model_path:
        print("\n💡 Available options:")
        print("• python eval_agent.py --list          # See all models")
        print("• python eval_agent.py --model best    # Use best model")
        print("• python eval_agent.py --model latest  # Use latest model")
        return
    
    print(f"🎯 Welcome to Agent Evaluation!")
    print(f"We're going to test how well your agent learned to manipulate the cube.")
    
    if args.render:
        print("📺 GUI mode: You'll see the robot in action!")
        print("⚠️  Note: Close the GUI window manually when done watching")
    else:
        print("🚀 Headless mode: Fast evaluation without GUI")
        print("💡 Add --render to see the robot in action")
    print()
    
    # Create evaluator
    evaluator = CubeEvaluator(model_path, args.render, args.episodes)
    
    # Run evaluation
    success = evaluator.run_evaluation()
    
    if success:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"💡 To improve performance, try:")
        print(f"• python train_agent.py --continue {model_path}")
        print(f"• python train_agent.py --long")
    else:
        print(f"\n💡 If evaluation failed, try:")
        print(f"• python train_agent.py --quick-test")
        print(f"• Check that the model file exists and is valid")

if __name__ == "__main__":
    main() 