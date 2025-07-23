import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class CubeManipulation:

    SIMULATION_STEP_DELAY = 1 / 1000.  # Further reduced for better stability

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        self.step_counter = 0  # Track steps without heavy progress bar
        self.camera = camera

        # define environment with better stability settings
        if self.vis:
            self.physicsClient = p.connect(p.GUI)
            # Configure GUI for better stability on macOS
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # Disable shadows for performance
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Set physics parameters for stability
        p.setTimeStep(1./240.)  # Standard physics timestep
        
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        # Load the smaller cube instead of the screw box
        self.cubeID = p.loadURDF("./urdf/objects/small_cube.urdf",
                                [0.1, 0.1, 0.02],  # Position cube slightly off-center, half-height above ground
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=False)  # Allow the cube to move freely

        # Target position for the cube (goal)
        self.target_position = np.array([0.0, -0.1, 0.01125])  # Target location for the cube
        
        # Task completion tracking
        self.task_completed = False
        self.initial_cube_position = None

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        try:
            p.stepSimulation()
            if self.vis:
                time.sleep(self.SIMULATION_STEP_DELAY)
                self.step_counter += 1
                # Print status occasionally instead of continuous progress bar
                if self.step_counter % 1000 == 0:
                    print(f"Simulation steps: {self.step_counter}")
        except p.error:
            # Connection lost - stop simulation
            raise RuntimeError("Physics simulation connection lost")

    def read_debug_parameter(self):
        # read the value of task parameter
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
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(20):  # Reduced from 120 to 20 for faster response
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
        """
        Simplified reward function for cube manipulation task.
        Reward is based on how close the cube is to the target position.
        """
        cube_pos = self.get_cube_position()
        distance_to_target = np.linalg.norm(cube_pos[:2] - self.target_position[:2])  # Only consider x,y distance
        
        # Success threshold
        success_threshold = 0.05  # 5cm tolerance
        
        if distance_to_target < success_threshold and not self.task_completed:
            self.task_completed = True
            print('Cube successfully moved to target position!')
            return 1.0
        
        # Continuous reward based on negative distance (closer is better)
        # Scale so that reward is between 0 and 1
        max_distance = 0.5  # Maximum expected distance
        continuous_reward = max(0, 1 - (distance_to_target / max_distance))
        
        return continuous_reward

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())
        
        # Add cube position to observation
        obs['cube_position'] = self.get_cube_position()
        obs['target_position'] = self.target_position

        return obs

    def reset_cube(self):
        """Reset cube to initial position"""
        initial_pos = [0.1, 0.1, 0.01125]  # Updated for smaller cube height
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.cubeID, initial_pos, initial_orientation)
        
        # Reset cube velocity
        p.resetBaseVelocity(self.cubeID, [0, 0, 0], [0, 0, 0])

    def reset(self):
        self.robot.reset()
        self.reset_cube()
        self.task_completed = False
        return self.get_observation()

    def is_connected(self):
        """Check if the physics server is still connected"""
        try:
            return p.isConnected(self.physicsClient)
        except:
            return False
    
    def close(self):
        try:
            if p.isConnected(self.physicsClient):
                p.disconnect(self.physicsClient)
        except:
            pass  # Already disconnected or connection error