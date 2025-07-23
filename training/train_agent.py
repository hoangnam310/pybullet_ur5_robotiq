#!/usr/bin/env python3
"""
Cube Manipulation Agent Training Script

This script trains a reinforcement learning agent to manipulate a cube
using a robotic arm. It's designed to be easy to use and understand!

Usage:
    python train_agent.py                    # Standard training
    python train_agent.py --quick-test       # Quick 5-minute test
    python train_agent.py --short            # 1-hour training session
    python train_agent.py --long             # Full training session
    python train_agent.py --continue MODEL   # Continue training from saved model
"""

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import RL libraries
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    print("✅ Stable-Baselines3 imported successfully!")
except ImportError:
    print("❌ Error: stable-baselines3 not installed!")
    print("Please install it with: pip install stable-baselines3[extra]")
    sys.exit(1)

# Import our modules
from config import TrainingConfig, ExperimentConfig, create_directories, detect_device, TRAINING_TIPS
from env_wrapper import CubeManipulationEnv, CubeManipulationEnvSimple

class CubeTrainer:
    """Main training class for the cube manipulation agent"""
    
    def __init__(self, experiment_type="standard", enable_gui=False):
        self.experiment_type = experiment_type
        self.start_time = datetime.now()
        self.enable_gui = enable_gui
        
        # Create necessary directories
        create_directories()
        
        # Detect and configure device (Mac M4 GPU support)
        self.device = detect_device()
        
        # Configure experiment parameters
        self.setup_experiment()
        
        print(f"🤖 Cube Manipulation Agent Trainer")
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔬 Experiment: {experiment_type}")
        print(f"💾 Device: {self.device}")
        print(f"📺 GUI: {'Enabled' if enable_gui else 'Disabled (faster training)'}")
        print(f"⏱️  Total timesteps: {self.total_timesteps:,}")
        print("=" * 50)
    
    def setup_experiment(self):
        """Configure training parameters based on experiment type"""
        base_config = TrainingConfig()
        
        if self.experiment_type == "quick-test":
            config = ExperimentConfig.QUICK_TEST
            self.env_name = base_config.ENV_NAME  # Use simple environment for quick test
        elif self.experiment_type == "short":
            config = ExperimentConfig.SHORT_TRAINING
            self.env_name = base_config.ENV_NAME
        elif self.experiment_type == "long":
            config = ExperimentConfig.LONG_TRAINING
            self.env_name = base_config.ENV_NAME_FULL  # Use full environment for long training
        else:  # standard
            config = {
                'total_timesteps': base_config.TOTAL_TIMESTEPS,
                'save_freq': base_config.SAVE_FREQ,
                'eval_freq': base_config.EVAL_FREQ,
                'eval_episodes': base_config.EVAL_EPISODES
            }
            self.env_name = base_config.ENV_NAME
        
        # Set training parameters
        self.total_timesteps = config['total_timesteps']
        self.save_freq = config['save_freq'] 
        self.eval_freq = config['eval_freq']
        self.eval_episodes = config['eval_episodes']
        
        # Model paths
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.model_name = f"{base_config.MODEL_NAME}_{self.experiment_type}_{timestamp}"
        self.model_dir = base_config.MODEL_DIR
        self.log_dir = base_config.LOG_DIR
        self.tensorboard_log = base_config.TENSORBOARD_LOG
    
    def create_environment(self, render_mode=None, is_eval=False):
        """Create the training environment"""
        print(f"🏗️  Creating environment: {self.env_name}")
        
        # Determine render mode
        if render_mode is None:
            if is_eval and TrainingConfig.ENABLE_GUI_EVAL:
                render_mode = 'human'
                print("📺 Enabling GUI for evaluation")
            elif self.enable_gui and TrainingConfig.RENDER_TRAINING:
                render_mode = 'human'
                print("📺 Enabling GUI for training (slower but visual)")
            else:
                render_mode = None
                print("🚀 Using headless mode for faster training")
        
        # Create the environment
        if self.env_name == "CubeManipulationSimple-v0":
            env = CubeManipulationEnvSimple(render_mode=render_mode)
        else:
            env = CubeManipulationEnv(render_mode=render_mode)
        
        # Wrap with Monitor for logging
        log_file = os.path.join(self.log_dir, f"training_{self.experiment_type}")
        env = Monitor(env, log_file)
        
        return env
    
    def create_model(self, env, continue_from=None):
        """Create or load the RL model"""
        if continue_from:
            print(f"📂 Loading model from: {continue_from}")
            model = PPO.load(continue_from, env=env, device=self.device)
            print("✅ Model loaded successfully!")
        else:
            print(f"🧠 Creating new PPO model...")
            config = TrainingConfig()
            
            # Configure logging
            logger = configure(
                folder=os.path.join(self.log_dir, self.model_name),
                format_strings=['stdout', 'csv', 'tensorboard']
            )
            
            # Update config with detected device
            ppo_config = config.PPO_CONFIG.copy()
            ppo_config['device'] = self.device
            
            model = PPO(
                config.POLICY_TYPE,
                env,
                tensorboard_log=self.tensorboard_log,
                **ppo_config
            )
            model.set_logger(logger)
            print(f"✅ Model created successfully on {self.device}!")
        
        return model
    
    def setup_callbacks(self, eval_env):
        """Setup training callbacks for evaluation and saving"""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_dir, f"best_{self.model_name}"),
            log_path=self.log_dir,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
            render=TrainingConfig.ENABLE_GUI_EVAL and self.enable_gui,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback  
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=os.path.join(self.model_dir, f"checkpoints_{self.model_name}"),
            name_prefix=self.model_name,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks
    
    def train(self, continue_from=None):
        """Main training loop"""
        try:
            # Create environments
            print("🏗️  Setting up training environment...")
            train_env = self.create_environment()
            eval_env = self.create_environment(is_eval=True)
            
            # Create model
            model = self.create_model(train_env, continue_from)
            
            # Setup callbacks
            callbacks = self.setup_callbacks(eval_env)
            
            # Start training
            print("\n🚀 Starting training...")
            print(f"📊 Monitor progress with: tensorboard --logdir {self.tensorboard_log}")
            print(f"💾 Models will be saved to: {self.model_dir}")
            print("-" * 50)
            
            start_time = time.time()
            
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                tb_log_name=self.model_name
            )
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, f"final_{self.model_name}")
            model.save(final_model_path)
            
            # Training complete
            elapsed_time = time.time() - start_time
            print("\n🎉 Training completed!")
            print(f"⏱️  Total training time: {elapsed_time/3600:.2f} hours")
            print(f"💾 Final model saved: {final_model_path}")
            print(f"🏆 Best model saved: {os.path.join(self.model_dir, f'best_{self.model_name}')}")
            
            # Clean up
            train_env.close()
            eval_env.close()
            
            return final_model_path
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            print("💾 Saving current model...")
            
            interrupted_path = os.path.join(self.model_dir, f"interrupted_{self.model_name}")
            model.save(interrupted_path)
            print(f"💾 Model saved: {interrupted_path}")
            
            return interrupted_path
        
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            print("🔍 Check the logs for more details")
            return None

def print_welcome_message():
    """Print a friendly welcome message"""
    print("🎯 Welcome to Cube Manipulation Agent Training!")
    print("=" * 60)
    print("This script will train an AI agent to manipulate a cube")
    print("using a robotic arm. The agent learns through trial and error.")
    print()
    print("What will happen:")
    print("1. 🏗️  Environment setup")
    print("2. 🧠 Neural network creation")
    print("3. 🚀 Training begins (this takes time!)")
    print("4. 📊 Progress monitoring")
    print("5. 💾 Model saving")
    print()
    print("Tips:")
    print("• Be patient - robot learning takes time!")
    print("• Monitor progress with TensorBoard")
    print("• Start with quick-test for first time")
    print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train cube manipulation agent")
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick 5-minute test')
    parser.add_argument('--short', action='store_true',
                       help='Run short 1-hour training')
    parser.add_argument('--long', action='store_true', 
                       help='Run full training session')
    parser.add_argument('--continue', dest='continue_model', type=str,
                       help='Continue training from saved model')
    parser.add_argument('--gui', action='store_true',
                       help='Enable GUI visualization during training (slower)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU usage (disable MPS/CUDA)')
    parser.add_argument('--tips', action='store_true',
                       help='Show training tips and exit')
    
    args = parser.parse_args()
    
    # Show tips
    if args.tips:
        print(TRAINING_TIPS)
        return
    
    # Handle device override
    if args.no_gpu:
        TrainingConfig.USE_MPS = False
        TrainingConfig.DEVICE = "cpu"
        print("🔧 GPU disabled, using CPU only")
    
    # Determine experiment type
    if args.quick_test:
        experiment_type = "quick-test"
    elif args.short:
        experiment_type = "short"
    elif args.long:
        experiment_type = "long"
    else:
        experiment_type = "standard"
    
    # Print welcome message
    print_welcome_message()
    
    # Show GPU status
    if args.gui:
        print("📺 GUI mode enabled - you'll see the robot training!")
        print("⚠️  Note: GUI mode is slower but great for watching progress")
    
    # Confirm before starting (except for quick test)
    if experiment_type != "quick-test":
        response = input("Ready to start training? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Create trainer and start training
    trainer = CubeTrainer(experiment_type, enable_gui=args.gui)
    final_model = trainer.train(continue_from=args.continue_model)
    
    if final_model:
        print(f"\n🎯 Next steps:")
        print(f"1. Test your agent: python eval_agent.py --model {final_model}")
        print(f"2. View training plots: tensorboard --logdir {trainer.tensorboard_log}")
        print(f"3. Continue training: python train_agent.py --continue {final_model}")
        print("\n🎉 Happy robot learning! 🤖")
    else:
        print("\n💡 Try running with --quick-test first to check everything works")

if __name__ == "__main__":
    main() 