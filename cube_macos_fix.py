"""
macOS-Specific Cube Manipulation Fix
This version addresses PyBullet GUI stability issues on macOS.
"""

import numpy as np
import time
import pybullet as p
import pybullet_data
from robot import UR5Robotiq85
from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict

class CubeManipulationMacOS:
    
    SIMULATION_STEP_DELAY = 1 / 1000.

    def __init__(self, robot, models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        self.step_counter = 0
        self.camera = camera

        # macOS-specific GUI initialization
        if self.vis:
            print("🍎 Initializing PyBullet GUI for macOS...")
            self.physicsClient = p.connect(p.GUI, options="--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0")
            
            # Wait for GUI to fully initialize
            time.sleep(2)
            
            # macOS-specific GUI configuration
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
                print("✅ GUI configuration successful")
            except Exception as e:
                print(f"⚠️  GUI configuration warning: {e}")
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(1./240.)
        
        self.planeID = p.loadURDF("plane.urdf")
        
        # Initialize robot
        print("🤖 Loading robot...")
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # Add debug sliders with delay for macOS
        print("🎛️  Creating debug sliders...")
        time.sleep(1)  # Give GUI time to settle
        
        try:
            self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
            self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
            self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
            self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
            self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
            self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
            self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)
            print("✅ Debug sliders created successfully")
        except Exception as e:
            print(f"❌ Failed to create debug sliders: {e}")
            raise

        # Load cube
        print("📦 Loading cube...")
        self.cubeID = p.loadURDF("./urdf/objects/small_cube.urdf",
                                [0.1, 0.1, 0.02],
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=False)

        self.target_position = np.array([0.0, -0.1, 0.02])
        self.task_completed = False
        self.initial_cube_position = None
        
        print("🎯 Environment setup complete!")

    def step_simulation(self):
        """Hook p.stepSimulation()"""
        try:
            p.stepSimulation()
            if self.vis:
                time.sleep(self.SIMULATION_STEP_DELAY)
                self.step_counter += 1
                if self.step_counter % 2000 == 0:
                    print(f"📊 Simulation steps: {self.step_counter}")
        except p.error:
            raise RuntimeError("Physics simulation connection lost")

    def read_debug_parameter(self):
        """Read debug parameter values"""
        try:
            if not p.isConnected(self.physicsClient):
                raise RuntimeError("Physics server disconnected")
                
            x = p.readUserDebugParameter(self.xin)
            y = p.readUserDebugParameter(self.yin)
            z = p.readUserDebugParameter(self.zin)
            roll = p.readUserDebugParameter(self.rollId)
            pitch = p.readUserDebugParameter(self.pitchId)
            yaw = p.readUserDebugParameter(self.yawId)
            gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

            return x, y, z, roll, pitch, yaw, gripper_opening_length
        except p.error:
            raise RuntimeError("Failed to read debug parameters - GUI window may have been closed")

    def step(self, action, control_method='joint'):
        """Environment step function"""
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(20):
            self.step_simulation()

        reward = self.update_reward()
        done = True if reward == 1 else False
        info = dict(task_completed=self.task_completed, cube_position=self.get_cube_position())
        return self.get_observation(), reward, done, info

    def get_cube_position(self):
        """Get the current position of the cube"""
        try:
            if not p.isConnected(self.physicsClient):
                raise RuntimeError("Physics server disconnected")
            cube_pos, _ = p.getBasePositionAndOrientation(self.cubeID)
            return np.array(cube_pos)
        except p.error:
            raise RuntimeError("Failed to get cube position - connection lost")

    def update_reward(self):
        """Calculate reward based on cube position"""
        cube_pos = self.get_cube_position()
        distance_to_target = np.linalg.norm(cube_pos[:2] - self.target_position[:2])
        
        success_threshold = 0.05
        
        if distance_to_target < success_threshold and not self.task_completed:
            self.task_completed = True
            print('🎉 Cube successfully moved to target position!')
            return 1.0
        
        max_distance = 0.5
        continuous_reward = max(0, 1 - (distance_to_target / max_distance))
        
        return continuous_reward

    def get_observation(self):
        """Get current observation"""
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())
        
        obs['cube_position'] = self.get_cube_position()
        obs['target_position'] = self.target_position

        return obs

    def reset_cube(self):
        """Reset cube to initial position"""
        initial_pos = [0.1, 0.1, 0.01125]
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.cubeID, initial_pos, initial_orientation)
        p.resetBaseVelocity(self.cubeID, [0, 0, 0], [0, 0, 0])

    def reset(self):
        """Reset environment"""
        self.robot.reset()
        self.reset_cube()
        self.task_completed = False
        return self.get_observation()

    def is_connected(self):
        """Check if physics server is connected"""
        try:
            return p.isConnected(self.physicsClient)
        except:
            return False
    
    def close(self):
        """Close environment"""
        try:
            if p.isConnected(self.physicsClient):
                p.disconnect(self.physicsClient)
        except:
            pass

def main():
    print("🍎 macOS PyBullet Cube Manipulation")
    print("=" * 50)
    
    # Initialize robot
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    models = Models()
    
    try:
        # Create environment with macOS fixes
        env = CubeManipulationMacOS(robot, models, vis=True)
        
        print("\n🔷 Environment Ready!")
        print("📱 If you can see the PyBullet GUI window with sliders, we've fixed the issue!")
        print("🎮 Use the debug sliders to control the robot")
        print("🚪 Close the GUI window or press Ctrl+C to exit")
        print("-" * 50)
        
        # Reset environment
        obs = env.reset()
        print(f"📍 Initial cube position: {obs['cube_position']}")
        print(f"🎯 Target position: {obs['target_position']}")
        print()
        
        # Main control loop
        step_count = 0
        last_report_time = time.time()
        
        # Give user time to see the GUI
        print("⏳ Keeping GUI stable for 5 seconds...")
        time.sleep(5)
        print("🚀 Starting control loop!")
        
        while True:
            try:
                # Check connection periodically
                if step_count > 0 and step_count % 3000 == 0:
                    if not env.is_connected():
                        print("⚠️  Connection lost - exiting")
                        break
                
                # Read control inputs
                x, y, z, roll, pitch, yaw, gripper = env.read_debug_parameter()
                action = [x, y, z, roll, pitch, yaw, gripper]
                
                # Take step
                obs, reward, done, info = env.step(action, control_method='end')
                step_count += 1
                
                # Time-based status reporting
                current_time = time.time()
                if current_time - last_report_time > 15.0:  # Every 15 seconds
                    cube_pos = obs['cube_position']
                    distance = np.linalg.norm(cube_pos[:2] - obs['target_position'][:2])
                    print(f"🔄 Steps: {step_count} | Distance: {distance:.3f} | Reward: {reward:.3f}")
                    last_report_time = current_time
                
                if done:
                    print("🎉 Task completed! Resetting...")
                    obs = env.reset()
                    step_count = 0
                    last_report_time = time.time()
                    
            except RuntimeError as e:
                if "disconnected" in str(e) or "closed" in str(e) or "parameter" in str(e):
                    print("\n📱 GUI window closed - exiting gracefully")
                    break
                else:
                    print(f"\n⚠️  Error: {e}")
                    break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                break
                
    except KeyboardInterrupt:
        print("\n⌨️  Keyboard interrupt - shutting down...")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("This might be a PyBullet/macOS compatibility issue")
    finally:
        print("\n🧹 Cleaning up...")
        try:
            env.close()
            print("✅ Cleanup successful!")
        except:
            print("⚠️  Environment already closed")

if __name__ == "__main__":
    main() 