"""
Gym Environment Wrapper for Cube Manipulation Training

This wrapper makes our cube environment compatible with reinforcement learning libraries
like Stable-Baselines3, OpenAI Gym, etc.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Add parent directory to path so we can import our cube environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cube_env import CubeManipulation
from robot import UR5Robotiq85
from utilities import Models


class CubeManipulationEnv(gym.Env):
    """
    OpenAI Gym wrapper for cube manipulation environment.
    
    The agent needs to learn to move a cube from its initial position to a target position
    using a robotic arm with a gripper.
    """
    
    def __init__(self, render_mode=None):
        super(CubeManipulationEnv, self).__init__()
        
        # Save current directory and change to parent directory for URDF loading
        self.original_cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(parent_dir)
        
        try:
            # Initialize the underlying environment
            self.robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
            self.models = Models()
            
            # Use headless mode for training (much faster)
            self.render_mode = render_mode
            vis_mode = (render_mode == 'human')
            
            self.env = CubeManipulation(self.robot, self.models, vis=vis_mode)
        finally:
            # Restore original directory
            os.chdir(self.original_cwd)
        
        # Define action space: [x, y, z, roll, pitch, yaw, gripper_opening]
        # All values are normalized between -1 and 1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )
        
        # Define observation space
        # We'll use: robot joint positions (6) + cube position (3) + target position (3) + distance to target (1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),  # 6 + 3 + 3 + 1 = 13 features
            dtype=np.float32
        )
        
        # Action scaling parameters (to convert from [-1,1] to actual ranges)
        self.action_scales = {
            'position': 0.224,  # Max reach of robot
            'orientation': np.pi,  # Full rotation range
            'gripper': 0.085  # Max gripper opening
        }
        
        # Tracking variables
        self.episode_length = 0
        self.max_episode_length = 1000  # Maximum steps per episode
        self.success_threshold = 0.05  # Distance threshold for success (5cm)
        
        print("🤖 Cube Manipulation RL Environment initialized!")
        print(f"   Action space: {self.action_space.shape}")
        print(f"   Observation space: {self.observation_space.shape}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the underlying environment
        obs = self.env.reset()
        
        # Reset episode tracking
        self.episode_length = 0
        
        # Convert to gym format
        gym_obs = self._get_gym_observation(obs)
        
        return gym_obs, {}
    
    def step(self, action):
        """Take a step in the environment"""
        # Scale actions from [-1,1] to actual ranges
        scaled_action = self._scale_action(action)
        
        # Take step in underlying environment
        obs, reward, done, info = self.env.step(scaled_action, control_method='end')
        
        # Update episode tracking
        self.episode_length += 1
        
        # Convert to gym format
        gym_obs = self._get_gym_observation(obs)
        
        # Check if episode should terminate
        terminated = done or (self.episode_length >= self.max_episode_length)
        truncated = (self.episode_length >= self.max_episode_length)
        
        # Add more info for logging
        info.update({
            'episode_length': self.episode_length,
            'cube_distance': np.linalg.norm(obs['cube_position'][:2] - obs['target_position'][:2]),
            'is_success': done
        })
        
        return gym_obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (already handled by vis=True in underlying env)"""
        pass
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def _scale_action(self, action):
        """Convert normalized action [-1,1] to actual robot commands"""
        scaled = np.zeros(7)
        
        # Position commands (x, y, z)
        scaled[0] = action[0] * self.action_scales['position']  # x
        scaled[1] = action[1] * self.action_scales['position']  # y  
        scaled[2] = action[2] * 0.5 + 0.5  # z (always positive, 0 to 1m)
        
        # Orientation commands (roll, pitch, yaw)
        scaled[3] = action[3] * self.action_scales['orientation']  # roll
        scaled[4] = action[4] * self.action_scales['orientation']  # pitch
        scaled[5] = action[5] * self.action_scales['orientation']  # yaw
        
        # Gripper command
        scaled[6] = (action[6] + 1) * 0.5 * self.action_scales['gripper']  # 0 to max_opening
        
        return scaled
    
    def _get_gym_observation(self, obs):
        """Convert environment observation to gym format"""
        # Get robot joint positions (6 joints)
        joint_positions = []
        for joint_name in ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']:
            if joint_name in obs:
                joint_positions.append(obs[joint_name])
            else:
                joint_positions.append(0.0)  # Fallback
        
        # Ensure we have exactly 6 joint positions
        while len(joint_positions) < 6:
            joint_positions.append(0.0)
        joint_positions = joint_positions[:6]
        
        # Get cube and target positions
        cube_pos = obs['cube_position'][:3]  # x, y, z
        target_pos = obs['target_position'][:3]  # x, y, z
        
        # Calculate distance to target
        distance = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        
        # Combine all observations
        gym_obs = np.concatenate([
            joint_positions,  # 6 features
            cube_pos,        # 3 features
            target_pos,      # 3 features
            [distance]       # 1 feature
        ])
        
        return gym_obs.astype(np.float32)


class CubeManipulationEnvSimple(gym.Env):
    """
    Simplified version with fewer observation features for faster training
    """
    
    def __init__(self, render_mode=None):
        super(CubeManipulationEnvSimple, self).__init__()
        
        # Save current directory and change to parent directory for URDF loading
        self.original_cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(parent_dir)
        
        try:
            # Initialize the underlying environment
            self.robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
            self.models = Models()
            
            vis_mode = (render_mode == 'human')
            self.env = CubeManipulation(self.robot, self.models, vis=vis_mode)
        finally:
            # Restore original directory
            os.chdir(self.original_cwd)
        
        # Simplified action space: just [x, y, gripper] - focus on 2D movement
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Simplified observation: cube position (2D) + target position (2D) + distance (1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),  # 2 + 2 + 1 = 5 features
            dtype=np.float32
        )
        
        self.episode_length = 0
        self.max_episode_length = 500  # Shorter episodes for faster training
        
        print("🎯 Simple Cube Manipulation RL Environment initialized!")
        print(f"   Action space: {self.action_space.shape} (x, y, gripper)")
        print(f"   Observation space: {self.observation_space.shape} (simplified)")
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        obs = self.env.reset()
        self.episode_length = 0
        gym_obs = self._get_simple_observation(obs)
        
        return gym_obs, {}
    
    def step(self, action):
        # Convert simplified action to full action
        full_action = np.array([
            action[0] * 0.224,  # x
            action[1] * 0.224,  # y
            0.3,                # z (fixed height)
            0.0,                # roll
            np.pi/2,           # pitch (down)
            np.pi/2,           # yaw
            (action[2] + 1) * 0.5 * 0.085  # gripper
        ])
        
        obs, reward, done, info = self.env.step(full_action, control_method='end')
        self.episode_length += 1
        
        gym_obs = self._get_simple_observation(obs)
        terminated = done or (self.episode_length >= self.max_episode_length)
        truncated = (self.episode_length >= self.max_episode_length)
        
        info.update({
            'episode_length': self.episode_length,
            'cube_distance': np.linalg.norm(obs['cube_position'][:2] - obs['target_position'][:2]),
            'is_success': done
        })
        
        return gym_obs, reward, terminated, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        self.env.close()
    
    def _get_simple_observation(self, obs):
        cube_pos = obs['cube_position'][:2]  # x, y only
        target_pos = obs['target_position'][:2]  # x, y only
        distance = np.linalg.norm(cube_pos - target_pos)
        
        gym_obs = np.concatenate([
            cube_pos,    # 2 features
            target_pos,  # 2 features  
            [distance]   # 1 feature
        ])
        
        return gym_obs.astype(np.float32)


# Register environments with gymnasium
from gymnasium.envs.registration import register

register(
    id='CubeManipulation-v0',
    entry_point='env_wrapper:CubeManipulationEnv',
    max_episode_steps=1000,
)

register(
    id='CubeManipulationSimple-v0', 
    entry_point='env_wrapper:CubeManipulationEnvSimple',
    max_episode_steps=500,
) 