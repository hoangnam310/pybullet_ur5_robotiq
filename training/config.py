"""
Training Configuration for Cube Manipulation Agent

This file contains all the hyperparameters and settings for training.
Feel free to experiment with these values to improve training performance!
"""

import os

class TrainingConfig:
    """Configuration class for training parameters"""
    
    # Environment settings
    ENV_NAME = "CubeManipulationSimple-v0"  # Start with simple version
    ENV_NAME_FULL = "CubeManipulation-v0"   # Full version for advanced training
    
    # GPU/Device settings for Mac M4 (Apple Silicon)
    DEVICE = "auto"  # auto, cpu, mps (Metal Performance Shaders for Apple Silicon)
    USE_MPS = True   # Enable Metal Performance Shaders on Apple Silicon
    
    # Training hyperparameters
    TOTAL_TIMESTEPS = 100000        # How long to train (start small, increase later)
    LEARNING_RATE = 3e-4           # How fast the agent learns (standard value)
    
    # Algorithm settings (using PPO - good for robotics)
    ALGORITHM = "PPO"              # Proximal Policy Optimization
    POLICY_TYPE = "MlpPolicy"      # Multi-layer perceptron policy
    
    # PPO specific parameters
    PPO_CONFIG = {
        'learning_rate': LEARNING_RATE,
        'n_steps': 2048,           # Steps to collect before updating
        'batch_size': 64,          # Training batch size
        'n_epochs': 10,            # Number of epochs to train on each batch
        'gamma': 0.99,             # Discount factor for future rewards
        'gae_lambda': 0.95,        # Lambda for GAE (Generalized Advantage Estimation)
        'clip_range': 0.2,         # Clipping parameter for PPO
        'ent_coef': 0.0,          # Entropy coefficient for exploration
        'vf_coef': 0.5,           # Value function coefficient
        'max_grad_norm': 0.5,      # Gradient clipping
        'verbose': 1,              # Logging level
        'device': DEVICE           # Use GPU/MPS if available
    }
    
    # Alternative algorithm configs (uncomment to try)
    # SAC_CONFIG = {
    #     'learning_rate': 3e-4,
    #     'buffer_size': 1000000,
    #     'learning_starts': 100,
    #     'batch_size': 256,
    #     'tau': 0.005,
    #     'gamma': 0.99,
    #     'train_freq': 1,
    #     'gradient_steps': 1,
    #     'verbose': 1
    # }
    
    # Training schedule
    SAVE_FREQ = 10000             # Save model every N steps
    EVAL_FREQ = 5000              # Evaluate model every N steps
    EVAL_EPISODES = 10            # Number of episodes for evaluation
    
    # Visualization settings
    ENABLE_GUI_EVAL = True        # Show GUI during evaluation
    RECORD_VIDEO = False          # Record training videos (experimental)
    RENDER_TRAINING = False      # Show GUI during training (much slower!)
    
    # File paths
    MODEL_DIR = "training/models"
    LOG_DIR = "training/logs"
    TENSORBOARD_LOG = "training/tensorboard"
    
    # Model names
    MODEL_NAME = "cube_manipulation_agent"
    BEST_MODEL_NAME = "cube_manipulation_best"
    
    # Training phases (progressive training strategy)
    PHASES = {
        "phase1": {
            "name": "Simple Learning",
            "env": "CubeManipulationSimple-v0",
            "timesteps": 50000,
            "description": "Learn basic cube pushing with simplified controls"
        },
        "phase2": {
            "name": "Full Control",
            "env": "CubeManipulation-v0", 
            "timesteps": 100000,
            "description": "Learn full 6DOF manipulation"
        }
    }

class ExperimentConfig:
    """Configuration for different experiments you can try"""
    
    # Experiment 1: Quick test (for debugging)
    QUICK_TEST = {
        'total_timesteps': 1000,
        'save_freq': 500,
        'eval_freq': 500,
        'eval_episodes': 3
    }
    
    # Experiment 2: Short training (for initial results)
    SHORT_TRAINING = {
        'total_timesteps': 50000,
        'save_freq': 5000,
        'eval_freq': 2500,
        'eval_episodes': 5
    }
    
    # Experiment 3: Long training (for best results)  
    LONG_TRAINING = {
        'total_timesteps': 500000,
        'save_freq': 25000,
        'eval_freq': 10000,
        'eval_episodes': 20
    }

class RewardConfig:
    """Configuration for reward shaping (to help the agent learn better)"""
    
    # Reward weights
    SUCCESS_REWARD = 100.0         # Big reward for completing the task
    DISTANCE_REWARD_SCALE = 10.0   # Reward for getting closer to target
    STEP_PENALTY = -0.1           # Small penalty for each step (encourages efficiency)
    
    # Distance thresholds
    SUCCESS_DISTANCE = 0.05        # How close to target counts as success (5cm)
    PROGRESS_DISTANCE = 0.10       # Distance improvement threshold for progress reward

def get_model_path(model_name, step=None):
    """Helper function to get model save path"""
    if step:
        return os.path.join(TrainingConfig.MODEL_DIR, f"{model_name}_{step}")
    return os.path.join(TrainingConfig.MODEL_DIR, model_name)

def detect_device():
    """Detect the best available device for training (Mac M4 GPU support)"""
    try:
        import torch
        
        if TrainingConfig.DEVICE == "auto":
            # For Mac M4 (Apple Silicon), use MPS if available
            if torch.backends.mps.is_available() and TrainingConfig.USE_MPS:
                device = "mps"
                print("🚀 Using Apple Silicon GPU (MPS) for training - much faster!")
            elif torch.cuda.is_available():
                device = "cuda"
                print("🚀 Using NVIDIA GPU (CUDA) for training")
            else:
                device = "cpu"
                print("⚠️  Using CPU for training - consider using GPU for faster training")
        else:
            device = TrainingConfig.DEVICE
            print(f"🔧 Using manually specified device: {device}")
        
        # Test device functionality
        if device == "mps":
            try:
                test_tensor = torch.tensor([1.0]).to(device)
                print(f"✅ MPS device test successful")
                return device
            except Exception as e:
                print(f"⚠️  MPS not working, falling back to CPU: {e}")
                return "cpu"
        
        return device
        
    except ImportError:
        print("⚠️  PyTorch not found, using CPU")
        return "cpu"

def create_directories():
    """Create necessary directories for training"""
    dirs = [
        TrainingConfig.MODEL_DIR,
        TrainingConfig.LOG_DIR, 
        TrainingConfig.TENSORBOARD_LOG
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

# Training tips and recommendations
TRAINING_TIPS = """
🎯 Training Tips for Better Results:

1. Start Small: Begin with the simple environment and shorter training times
2. Monitor Progress: Watch the reward curves in TensorBoard
3. Adjust Learning Rate: If learning is too slow/fast, modify LEARNING_RATE
4. Experiment with Rewards: Try different reward shaping in RewardConfig
5. Progressive Training: Start with phase1, then continue with phase2
6. Save Often: Model training can take hours, save frequently!
7. Use GPU: If available, training will be much faster
8. Patience: Robot learning takes time, don't expect immediate results

🚀 Quick Start Commands:
- python train_agent.py --quick-test        # 5-minute test run
- python train_agent.py --short             # 1-hour training
- python train_agent.py --long              # Full training session
- python eval_agent.py --model best         # Test your trained agent
""" 