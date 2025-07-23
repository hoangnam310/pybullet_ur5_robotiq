#!/usr/bin/env python3
"""
Quick Setup Script for Cube Manipulation Training

This script helps you get started with training your robot quickly!
Run this first to make sure everything is set up correctly.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is suitable"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required. You have:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'stable_baselines3', 'gymnasium', 'tensorboard'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
        else:
            print(f"✅ {package} is installed")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'logs', 'tensorboard']
    
    for dir_name in dirs:
        dir_path = os.path.join('training', dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created: {dir_path}")

def run_quick_test():
    """Ask user if they want to run a quick test"""
    print("\n🎯 Setup complete! Ready to train your robot?")
    print("Options:")
    print("1. Run quick test (5 minutes)")
    print("2. Skip to manual training")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting quick test...")
        os.chdir('training')
        subprocess.run([sys.executable, 'train_agent.py', '--quick-test'])
    else:
        print("\n💡 To start training manually:")
        print("cd training/")
        print("python train_agent.py --quick-test")

def main():
    """Main setup function"""
    print("🤖 Cube Manipulation Training Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies  
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n💡 Install dependencies first:")
        print("cd training/")
        print("pip install -r requirements.txt")
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Run quick test
    run_quick_test()
    
    print("\n🎉 Setup complete! Happy robot training! 🤖")

if __name__ == "__main__":
    main() 