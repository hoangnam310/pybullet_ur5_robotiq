# 🤖 Cube Manipulation Agent Training

Welcome! This folder contains everything you need to train an AI agent to manipulate a cube using a robotic arm. Think of it as teaching a robot to learn through trial and error, just like how we humans learn new skills!

## 🎯 What This Does

Your AI agent will learn to:
- **Pick up** the cube with the robotic gripper
- **Move** it around in 3D space  
- **Place** it at a target location
- **Improve** its performance over time through practice

It's like watching a robot learn to play with blocks, but with real physics and smart AI!

## 🚀 Quick Start (For the Impatient)

Just want to see it work? Run this:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run a quick 5-minute test
python train_agent.py --quick-test

# 3. Test your trained agent
python eval_agent.py --model best --render
```

## 📋 Step-by-Step Guide

### Step 1: Install Dependencies

First, let's get all the required software:

```bash
pip install -r requirements.txt
```

This installs:
- **Stable-Baselines3**: The AI training library (think of it as the robot's brain)
- **TensorBoard**: For watching training progress (like a fitness tracker for AI)
- **Gym**: The standardized environment interface (like a video game API for robots)
- **PyTorch**: The neural network engine (the actual "neurons" of the AI)

### Step 2: Choose Your Training Adventure

We have different training options, like difficulty levels in a video game:

#### 🧪 **Quick Test** (5 minutes - Perfect for first time)
```bash
python train_agent.py --quick-test
```
- **What it does**: Tests that everything works
- **Time**: ~5 minutes
- **Good for**: Making sure your setup is correct

#### ⚡ **Short Training** (1 hour - Good results)
```bash
python train_agent.py --short
```
- **What it does**: Trains a reasonably good agent
- **Time**: ~1 hour  
- **Good for**: Getting your first working robot

#### 🎯 **Long Training** (4-8 hours - Best results)
```bash
python train_agent.py --long
```
- **What it does**: Trains a highly skilled agent
- **Time**: 4-8 hours
- **Good for**: Competition-ready performance

#### 📺 **Visual Training** (Watch your robot learn!)
```bash
python train_agent.py --short --gui
```
- **What it does**: Shows GUI during training so you can watch
- **Time**: Longer (GUI is slower)
- **Good for**: Understanding what your robot is doing

#### 🚀 **GPU Acceleration** (Mac M4 users!)
```bash
python train_agent.py --long  # Automatically uses Apple Silicon GPU
python train_agent.py --long --no-gpu  # Force CPU if needed
```
- **Automatic**: Uses your Mac M4 GPU automatically for 3-5x speedup
- **Manual**: Add `--no-gpu` to disable if you have issues

#### 🔄 **Continue Training** (Improve existing agent)
```bash
python train_agent.py --continue path/to/your/model
```
- **What it does**: Makes an existing agent even better
- **Time**: As long as you want
- **Good for**: Fine-tuning performance

### Step 3: Monitor Your Agent's Learning

While training, you can watch your AI learn in real-time! Open a new terminal and run:

```bash
tensorboard --logdir training/tensorboard
```

Then open http://localhost:6006 in your browser. You'll see:
- **Reward curves**: How well your agent is doing (higher = better)
- **Episode length**: How efficiently it solves the task
- **Success rate**: Percentage of successful attempts

Think of it like watching your robot's report card update in real time!

### Step 4: Test Your Trained Agent

Once training finishes, see how your agent performs:

```bash
# Test with GUI visualization (watch your robot work!)
python eval_agent.py --model best --render --episodes 10

# Quick evaluation without GUI (faster)
python eval_agent.py --model best --episodes 20

# Test on CPU only (if GPU issues)
python eval_agent.py --model best --render --no-gpu

# List all your trained models
python eval_agent.py --list
```

## 📁 What's in This Folder?

Here's what each file does (so you know what you're working with):

- **`train_agent.py`** 🧠 - The main training script (like a gym for your AI)
- **`eval_agent.py`** 🔬 - Tests how good your trained agent is (like final exams)
- **`env_wrapper.py`** 🏗️ - Connects our cube environment to AI libraries (like a translator)
- **`config.py`** ⚙️ - All the training settings and parameters (like difficulty settings)
- **`requirements.txt`** 📦 - List of software needed (like a shopping list)
- **`README.md`** 📖 - This guide you're reading!

## 🎛️ Customizing Your Training

Want to experiment? You can modify these settings in `config.py`:

### Training Duration
```python
TOTAL_TIMESTEPS = 100000  # More steps = better learning (but longer time)
```

### Learning Speed
```python
LEARNING_RATE = 3e-4  # Higher = learns faster but less stable
```

### Reward Shaping (How to motivate your robot)
```python
SUCCESS_REWARD = 100.0      # Big reward for completing task
DISTANCE_REWARD_SCALE = 10.0  # Reward for getting closer
STEP_PENALTY = -0.1         # Small penalty for taking too long
```

## 🚨 Troubleshooting

### "Import Error" or "Module Not Found"
```bash
# Make sure you're in the right directory
cd training/

# Reinstall dependencies
pip install -r requirements.txt

# If still problems, try creating a new virtual environment:
python -m venv training-env
source training-env/bin/activate  # On Windows: training-env\Scripts\activate
pip install -r requirements.txt
```

### Training is Too Slow
- **Mac M4 Users**: GPU acceleration is automatic! You should see "🚀 Using Apple Silicon GPU (MPS)"
- **Force CPU**: Add `--no-gpu` if you have GPU issues
- **Reduce timesteps**: Start with `--quick-test` or `--short`
- **Close other programs**: Free up RAM and CPU
- **Disable GUI**: Remove `--gui` flag for faster training

### Agent Isn't Learning Well
- **Train longer**: Try `--long` instead of `--short`
- **Check rewards**: Use TensorBoard to see if rewards are improving
- **Adjust parameters**: Experiment with learning rate in `config.py`

### GUI Window Closes Immediately (macOS issue)
This is a known PyBullet issue on macOS. The training will work headlessly (without visualization), which is actually faster anyway!

### GPU vs CPU Performance
- **Mac M4 GPU**: 3-5x faster training, automatic detection
- **CPU Only**: Slower but more compatible
- **Check Status**: Look for "🚀 Using Apple Silicon GPU (MPS)" message
- **GPU Issues**: Add `--no-gpu` to force CPU mode

## 📊 Understanding Results

After evaluation, you'll see statistics like:

- **Success Rate**: What percentage of attempts succeeded (aim for >70%)
- **Average Reward**: How well the agent performed overall
- **Episode Length**: How efficiently it solves the task (shorter = better)

### Performance Levels:
- **🌟 80%+ Success**: Excellent! Your robot is ready for the real world
- **👍 60-80% Success**: Good performance, could use some fine-tuning  
- **📚 30-60% Success**: Learning well, needs more training time
- **🔧 <30% Success**: Needs work - try longer training or different parameters

## 🎯 Advanced Tips

### Progressive Training Strategy
1. Start with `--quick-test` to verify everything works
2. Move to `--short` for your first real agent
3. Use `--continue` to improve your best model
4. Try `--long` for competition-ready performance

### Monitoring Best Practices
- Always check TensorBoard during training
- Save models frequently (done automatically)
- Test intermediate models to see progress

### Experimentation Ideas
- Try different reward functions in `config.py`
- Modify the environment wrapper for new challenges
- Experiment with different neural network architectures

## 🤝 Getting Help

If you're stuck:

1. **Check the console output** - it usually tells you what's wrong
2. **Look at TensorBoard** - see if training is progressing
3. **Try `--quick-test`** - verify basic functionality
4. **Read the error messages** - they're usually helpful!

## 🏆 Success Stories

What to expect:
- **After 5 minutes** (quick-test): Agent moves randomly but environment works
- **After 1 hour** (short): Agent consistently gets close to the target
- **After 4+ hours** (long): Agent reliably completes the task

Remember: Teaching a robot is like teaching a child - it takes patience, but the results are worth it!

## 🚀 Next Steps

Once you have a working agent:
1. **Experiment** with different reward functions
2. **Try more complex tasks** (multiple cubes, obstacles, etc.)
3. **Deploy** your agent on a real robot (if you have access)
4. **Share** your results with the robotics community!

---

**Happy robot training! 🤖✨**

*Remember: Every expert was once a beginner. Your first agent might not be perfect, but each training session makes it better. Be patient, experiment, and have fun watching your AI learn!* 