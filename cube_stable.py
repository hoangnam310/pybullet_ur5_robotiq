"""
Stable Cube Manipulation Example
This version is optimized for stability on macOS and reduces GUI crashes.
"""

import numpy as np
import time
from cube_env import CubeManipulation
from robot import UR5Robotiq85
from utilities import Models

def main():
    # Initialize robot
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    
    # Initialize models (if needed)
    models = Models()
    
    # Create the cube manipulation environment
    env = CubeManipulation(robot, models, vis=True)
    
    print("🔷 Stable Cube Manipulation Environment")
    print("Optimized for macOS stability")
    print("Task: Move the blue cube to the target position")
    print("Use debug sliders to control the robot")
    print("This version has improved stability and should not crash")
    print("Close the GUI window or press Ctrl+C to exit")
    print("-" * 50)
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"✅ Environment initialized successfully")
        print(f"📍 Initial cube position: {obs['cube_position']}")
        print(f"🎯 Target position: {obs['target_position']}")
        print()
        
        # Control loop with better stability
        step_count = 0
        last_report_time = time.time()
        
        while True:
            try:
                # Check connection every 2000 steps (less frequent checking)
                if step_count > 0 and step_count % 2000 == 0:
                    if not env.is_connected():
                        print("⚠️  Connection lost - exiting gracefully")
                        break
                
                # Read manual control inputs from debug sliders
                x, y, z, roll, pitch, yaw, gripper = env.read_debug_parameter()
                action = [x, y, z, roll, pitch, yaw, gripper]
                
                # Take a step in the environment
                obs, reward, done, info = env.step(action, control_method='end')
                
                step_count += 1
                
                # Time-based reporting instead of step-based (more stable)
                current_time = time.time()
                if current_time - last_report_time > 10.0:  # Every 10 seconds
                    cube_pos = obs['cube_position']
                    distance = np.linalg.norm(cube_pos[:2] - obs['target_position'][:2])
                    print(f"⏱️  Running for {int(current_time - last_report_time)} seconds | Steps: {step_count} | Distance: {distance:.3f} | Reward: {reward:.3f}")
                    last_report_time = current_time
                
                if done:
                    print("🎉 Task completed successfully!")
                    print(f"Final reward: {reward}")
                    print("Resetting for another attempt...")
                    obs = env.reset()
                    step_count = 0
                    last_report_time = time.time()
                    
            except RuntimeError as e:
                if "disconnected" in str(e) or "closed" in str(e) or "parameter" in str(e):
                    print("\n📱 GUI window was closed - exiting gracefully")
                    break
                else:
                    print(f"\n⚠️  Simulation error: {e}")
                    print("This might be due to GUI instability - trying to continue...")
                    time.sleep(1)  # Brief pause before continuing
                    break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n⌨️  Keyboard interrupt - shutting down...")
    finally:
        print("\n🧹 Cleaning up environment...")
        try:
            env.close()
            print("✅ Environment closed successfully!")
        except:
            print("⚠️  Environment was already closed")

if __name__ == "__main__":
    main() 