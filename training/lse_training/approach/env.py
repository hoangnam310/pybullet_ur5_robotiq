"""
Gym Environment Wrapper for Cube Approach Task
This inherits from the CubeManipulationEnv class in training/training_env.py
"""
import sys
import os

# Fix import path - go up to the main project directory FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p

# Import the training environment
from training.training_env import CubeManipulationEnv

class ApproachEnv(CubeManipulationEnv):
    """
    Environment for the approach task - getting gripper close to object
    Only controls end-effector position, orientation is fixed
    """
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        # Define max_episode_length based on the paper
        self.max_episode_length = 300
        # Approach-specific parameters
        self.approach_distance_threshold = 0.10  # 10cm from object (relaxed from 5cm)
        self.pre_grasp_height = 0.1  # Height above object for pre-grasp
        self.too_far_threshold = 0.8  # 80cm threshold for early termination (relaxed)
        
        # Action scaling parameters
        self.position_scale = 0.03  # 3cm max movement per step (even more precise)
        self.orientation_scale = 0.1  # 0.1 radian max rotation per step (more precise)
        
        # Action smoothing
        self.previous_action = None
        self.action_smoothing_factor = 0.7  # Smooth actions (70% previous + 30% new)
        
        # Collision detection parameters
        self.floor_height = 0.0  # Floor height for collision detection
        self.min_gripper_height = 0.02  # Minimum height above floor (2cm)
        self.min_wrist_height = 0.05  # Minimum height for wrist joints (5cm)
        self.collision_penalty = -1.0  # Very small penalty for gripper collisions
        self.self_collision_penalty = -2.0  # Small penalty for self-collision
        self.body_collision_penalty = -2.0  # Small penalty for body collisions
        self.min_approach_height = 0.05  # Reduced minimum height for proper approach
        
        # Progressive collision penalty system
        self.collision_penalty_start = -0.1  # Start with very small penalty
        self.collision_penalty_end = -5.0    # End with larger penalty  # Gradually increase over 50k steps
        self.total_training_steps = None  # Will be set by train.py
        self.current_training_steps = 0  # Current training step counter
        
        # Orientation reward parameters
        self.orientation_reward_scale = 2.0  # Scale factor for orientation rewards
        self.enable_orientation_reward = True  # Whether to include orientation rewards
        
        # Fixed orientation for approach (using current default)
        self.fixed_orientation = [0.0, np.pi/2, np.pi/2]  # [roll, pitch, yaw]
        
        # Approach-specific action space: [x, y, z, roll, pitch, yaw] - full 6 DOF control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32  # x, y, z, roll, pitch, yaw
        )
        
        # Approach-specific observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32  # [gripper_pos(3), cube_pos(3), distance(1), height_diff(1), wrist_1, wrist_2, wrist_3(3)]
        )
        
        print("🎯 Approach Environment initialized!")
        print(f"   Position scale: {self.position_scale}m per step")
        print(f"   Orientation scale: {self.orientation_scale}rad per step")
        print(f"   Approach threshold: {self.approach_distance_threshold}m")
        print(f"   Too far threshold: {self.too_far_threshold}m")
        print(f"   Min gripper height: {self.min_gripper_height}m")
        print(f"   Min approach height: {self.min_approach_height}m")
        print(f"   Gripper collision penalty: {self.collision_penalty}")
        print(f"   Self-collision penalty: {self.self_collision_penalty}")
        print(f"   Body collision penalty: {self.body_collision_penalty}")
        print(f"   Orientation reward scale: {self.orientation_reward_scale}")
        print(f"   Max episode length: {self.max_episode_length} steps")
    
    def _get_approach_observation(self, obs):
        """Get observation relevant for approach phase"""
        # obs is a numpy array from parent environment
        # Format: [joint_positions(6), ee_pos(3), cube_pos(3), target_pos(3), distance(1)]
        
        # Extract positions from the numpy array
        joint_positions = obs[:6]  # First 6 elements are joint positions
        ee_pos = obs[6:9]          # Next 3 elements are end-effector position
        cube_pos = obs[9:12]       # Next 3 elements are cube position
        
        # Use the actual end-effector position from the observation
        gripper_pos = ee_pos
        
        # Calculate distance to cube
        distance_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        

        
        # Height difference
        height_diff = gripper_pos[2] - cube_pos[2]
        
        # Get orientation information from joint positions (approximate)
        # Use wrist joints to approximate gripper orientation
        wrist_1 = joint_positions[3]  # wrist_1_joint
        wrist_2 = joint_positions[4]  # wrist_2_joint  
        wrist_3 = joint_positions[5]  # wrist_3_joint
        
        # Approach observation: [gripper_x, gripper_y, gripper_z, cube_x, cube_y, cube_z, distance, height_diff, wrist_1, wrist_2, wrist_3]
        approach_obs = np.concatenate([
            gripper_pos,      # 3 features
            cube_pos,         # 3 features
            [distance_to_cube], # 1 feature
            [height_diff],    # 1 feature
            [wrist_1, wrist_2, wrist_3]  # 3 features - orientation info
        ])
        
        return approach_obs.astype(np.float32)
    
    def _get_progressive_collision_penalty(self):
        """Calculate progressive collision penalty based on training progress"""
        # Handle case where total_training_steps is not set (e.g., during testing)
        if self.total_training_steps is None:
            return self.collision_penalty_start  # Use start penalty during testing
        
        # Linear interpolation from start to end penalty
        progress = min(1.0, self.current_training_steps / self.total_training_steps)
        current_penalty = self.collision_penalty_start + progress * (self.collision_penalty_end - self.collision_penalty_start)
        return current_penalty
    
    def get_end_effector_orientation(self):
        """Get the current end-effector orientation for debugging"""
        try:
            ee_state = p.getLinkState(self.env.robot.id, self.env.robot.eef_id)
            ee_orientation = ee_state[1]  # Quaternion [x, y, z, w]
            
            # Convert quaternion to euler angles for easier interpretation
            euler_angles = p.getEulerFromQuaternion(ee_orientation)
            return {
                'quaternion': ee_orientation,
                'euler': euler_angles,  # [roll, pitch, yaw]
                'raw_state': ee_state
            }
        except:
            return None
    
    def debug_collision_detection(self):
        """Debug collision detection to understand what's happening"""
        try:
            # Get all contact points with floor
            floor_contacts = p.getContactPoints(bodyA=self.env.robot.id, bodyB=self.env.planeID)
            
            # Get self-collision contacts
            self_contacts = p.getContactPoints(bodyA=self.env.robot.id, bodyB=self.env.robot.id)
            
            # Get gripper links
            gripper_links = self._get_gripper_link_ids()
            
            print(f"🔍 Collision Debug:")
            print(f"   Floor contacts: {len(floor_contacts)}")
            print(f"   Self-collision contacts: {len(self_contacts)}")
            print(f"   Gripper link IDs: {gripper_links}")
            
            if len(floor_contacts) > 0:
                print(f"   Floor contact details:")
                for i, contact in enumerate(floor_contacts):
                    link_a = contact[3]  # Robot link
                    link_b = contact[4]  # Floor
                    contact_pos = contact[5]  # Contact position
                    contact_depth = contact[8]  # Penetration depth
                    
                    link_type = "GRIPPER" if link_a in gripper_links else "BODY"
                    print(f"     Contact {i}: Robot link {link_a} ({link_type}) -> Floor, pos={contact_pos}, depth={contact_depth:.4f}")
            
            if len(self_contacts) > 0:
                print(f"   Self-collision details:")
                for i, contact in enumerate(self_contacts):
                    link_a = contact[3]  # Link A
                    link_b = contact[4]  # Link B
                    contact_pos = contact[5]  # Contact position
                    contact_depth = contact[8]  # Penetration depth
                    
                    if link_a != link_b:  # Only show different links
                        print(f"     Contact {i}: Link {link_a} -> Link {link_b}, pos={contact_pos}, depth={contact_depth:.4f}")
            
            # Check gripper height
            ee_state = p.getLinkState(self.env.robot.id, self.env.robot.eef_id)
            ee_pos = ee_state[0]
            print(f"   End-effector height: {ee_pos[2]:.4f}m")
            print(f"   Floor collision threshold: {self.floor_height + 0.01:.4f}m")
            
            # Test collision detection methods
            floor_collision = self._check_floor_collision(ee_pos)
            body_collision = self._check_body_collision()
            self_collision = self._check_self_collision()
            
            print(f"   Collision detection results:")
            print(f"     Floor collision: {floor_collision}")
            print(f"     Body collision: {body_collision}")
            print(f"     Self collision: {self_collision}")
            
            return {
                'floor_contacts': floor_contacts,
                'self_contacts': self_contacts,
                'gripper_links': gripper_links,
                'ee_height': ee_pos[2],
                'floor_collision': floor_collision,
                'body_collision': body_collision,
                'self_collision': self_collision
            }
        except Exception as e:
            print(f"   Error in collision debug: {e}")
            return None
    
    def _check_floor_collision(self, gripper_pos):
        """Check if gripper is colliding with the floor using height-based detection"""
        # Simple and reliable height-based collision detection
        # Floor is at height 0.0, gripper should stay above min_gripper_height
        return gripper_pos[2] < self.min_gripper_height
    
    def _check_self_collision(self):
        """Check if robot is colliding with itself"""
        try:
            # Get contact points between robot links (self-collision)
            contact_points = p.getContactPoints(bodyA=self.env.robot.id, bodyB=self.env.robot.id)
            
            # Check if any contact is between different robot links (self-collision)
            for contact in contact_points:
                link_a = contact[3]  # Link A
                link_b = contact[4]  # Link B
                
                # If both links belong to the robot and are different, it's self-collision
                if link_a != -1 and link_b != -1 and link_a != link_b:  # -1 means base link
                    return True
            
            return False
        except:
            # If collision detection fails, assume no collision
            return False
    
    def _check_body_collision(self):
        """Check if robot body (non-gripper parts) is colliding with floor"""
        # Use PyBullet contact detection for more accurate body collision detection
        try:
            # Get contact points between robot and floor
            contact_points = p.getContactPoints(bodyA=self.env.robot.id, bodyB=self.env.planeID)
            
            # Get gripper links
            gripper_links = self._get_gripper_link_ids()
            
            # Check if any contact involves non-gripper links
            for contact in contact_points:
                link_a = contact[3]  # Robot link
                link_b = contact[4]  # Floor (should be -1 for plane)
                
                # If the contact involves a link that's not gripper and not base, it's a body collision
                if link_a not in gripper_links and link_a != 0 and link_a != -1:
                    return True
            
            return False
        except:
            # Fallback: simple height-based detection
            try:
                # Get all robot link positions
                robot_links = []
                for i in range(p.getNumJoints(self.env.robot.id)):
                    link_state = p.getLinkState(self.env.robot.id, i)
                    link_pos = link_state[0]
                    robot_links.append((i, link_pos))
                
                # Get gripper links
                gripper_links = self._get_gripper_link_ids()
                
                # Check if any non-gripper link is too close to floor
                for link_id, link_pos in robot_links:
                    if link_id not in gripper_links and link_id != 0:  # Not gripper and not base
                        if link_pos[2] < 0.05:  # 5cm above floor threshold (more sensitive)
                            return True
                
                return False
            except:
                return False
    
    def _get_gripper_link_ids(self):
        """Get all gripper-related link IDs more reliably"""
        gripper_links = [self.env.robot.eef_id]  # Main end-effector link
        
        # Try to find gripper finger links by checking joint names
        try:
            for joint in self.env.robot.joints:
                joint_name = joint.name.lower()
                # Look for gripper-related joint names
                if any(keyword in joint_name for keyword in ['finger', 'gripper', 'hand', 'claw']):
                    gripper_links.append(joint.id)
        except:
            pass
        
        # Fallback: try sequential links after end-effector
        try:
            for i in range(self.env.robot.eef_id + 1, self.env.robot.eef_id + 10):
                try:
                    link_state = p.getLinkState(self.env.robot.id, i)
                    gripper_links.append(i)
                except:
                    break
        except:
            pass
        
        return gripper_links
    
    def _check_collisions(self, gripper_pos):
        """Check for all types of collisions"""
        floor_collision = self._check_floor_collision(gripper_pos)
        self_collision = self._check_self_collision()
        body_collision = self._check_body_collision()
        
        return {
            'floor_collision': floor_collision,
            'self_collision': self_collision,
            'body_collision': body_collision,
            'any_collision': floor_collision or self_collision or body_collision
        }
    
    def _compute_approach_reward(self, obs, action):
        """Compute dense reward with improved orientation and positioning rewards"""
        # obs is the approach observation format: [gripper_x, gripper_y, gripper_z, cube_x, cube_y, cube_z, distance, height_diff, wrist_1, wrist_2, wrist_3]
        
        # Extract positions from the approach observation
        gripper_pos = obs[:3]      # First 3 elements are gripper position
        cube_pos = obs[3:6]        # Next 3 elements are cube position
        distance = obs[6]          # Distance is already computed
        height_diff = obs[7]       # Height difference is already computed
        
        # Base distance reward: rt = -d(agt - sgt)
        reward = -distance  # rt = -d(agt - sgt)
        
        # Additional distance-based rewards for better learning
        if distance < 0.3:  # Within 30cm
            reward += 2.0
        if distance < 0.2:  # Within 20cm  
            reward += 3.0
        if distance < 0.15:  # Within 15cm
            reward += 5.0
        if distance < 0.1:   # Within 10cm
            reward += 10.0
        
        # Wrist height reward - encourage keeping wrist joints above floor
        try:
            # Get wrist joint positions
            wrist_1_state = p.getLinkState(self.env.robot.id, 4)  # wrist_1_joint
            wrist_2_state = p.getLinkState(self.env.robot.id, 5)  # wrist_2_joint
            wrist_3_state = p.getLinkState(self.env.robot.id, 6)  # wrist_3_joint
            
            wrist_1_height = wrist_1_state[0][2]
            wrist_2_height = wrist_2_state[0][2]
            wrist_3_height = wrist_3_state[0][2]
            
            # Reward for keeping wrists above minimum height
            min_wrist_height = min(wrist_1_height, wrist_2_height, wrist_3_height)
            if min_wrist_height > self.min_wrist_height:
                reward += 1.0  # Small reward for good wrist positioning
            else:
                reward -= 2.0  # Penalty for low wrist position
        except:
            pass  # Skip if we can't get wrist positions
        
        # Enhanced orientation reward: encourage gripper to point downward toward cube
        if self.enable_orientation_reward:
            orientation_reward = self._compute_orientation_reward(gripper_pos, cube_pos)
            reward += orientation_reward
        
        # Position-based rewards
        horizontal_distance = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])
        
        # Reward for being above the cube (height-based)
        if height_diff > 0.02:  # At least 2cm above cube
            height_reward = min(5.0, height_diff * 10)  # Up to 5 points for being above
            reward += height_reward
        
        # Reward for being close horizontally (horizontal positioning)
        if horizontal_distance < 0.1:  # Within 10cm horizontally
            horizontal_reward = (0.1 - horizontal_distance) * 20  # Up to 2 points for close horizontal positioning
            reward += horizontal_reward
        
        # Enhanced success bonus: +50 points when approach is successful
        # Success criteria: close distance + above cube + pointing down
        if (distance < self.approach_distance_threshold and 
            height_diff > 0.02 and 
            horizontal_distance < 0.05 and  # Very close horizontally (5cm)
            self._check_downward_orientation()):
            reward += 50  # Much higher success bonus
        
        # Pose quality reward - encourage natural robot poses
        try:
            # Get joint positions and check for extreme angles
            joint_positions = obs[8:11]  # wrist joints from observation
            wrist_1, wrist_2, wrist_3 = joint_positions
            
            # Penalize extreme joint angles (unnatural poses)
            extreme_angle_penalty = 0
            if abs(wrist_1) > 2.5:  # More than 2.5 radians
                extreme_angle_penalty += 1.0
            if abs(wrist_2) > 2.5:
                extreme_angle_penalty += 1.0
            if abs(wrist_3) > 2.5:
                extreme_angle_penalty += 1.0
            
            reward -= extreme_angle_penalty
        except:
            pass
        
        # Bonus for being in the "sweet spot" (5cm above, pointing down)
        if (0.03 < height_diff < 0.07 and  # Between 3-7cm above
            horizontal_distance < 0.03 and  # Very close horizontally (3cm)
            self._check_downward_orientation()):
            reward += 20  # Sweet spot bonus
        
        return reward
    
    def _check_proper_approach_pose(self, gripper_pos, cube_pos):
        """Check if the robot is in a proper approach pose (above the cube)"""
        # Check if gripper is above the cube
        height_diff = gripper_pos[2] - cube_pos[2]
        
        # Check horizontal distance
        horizontal_distance = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])
        
        # Proper approach: gripper should be above cube with reasonable horizontal distance
        is_above = height_diff > self.min_approach_height
        is_close_horizontally = horizontal_distance < 0.2  # 20cm horizontal tolerance
        
        return is_above and is_close_horizontally
    
    def _check_downward_orientation(self):
        """Check if gripper is pointing downward (negative z-axis)"""
        try:
            # Get end-effector orientation from PyBullet
            ee_state = p.getLinkState(self.env.robot.id, self.env.robot.eef_id)
            ee_orientation = ee_state[1]  # Quaternion [x, y, z, w]
            
            # Convert quaternion to rotation matrix
            rotation_matrix = p.getMatrixFromQuaternion(ee_orientation)
            rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
            
            # Get the z-axis of the end-effector (gripper direction)
            gripper_z_axis = rotation_matrix[:, 2]  # Third column is z-axis
            
            # Check if gripper is pointing downward (negative z in world coordinates)
            # We want the z-component to be negative (pointing down)
            pointing_down = gripper_z_axis[2] < -0.3  # At least 30 degrees downward
            
            return pointing_down
            
        except Exception as e:
            print(f"Warning: Could not check orientation: {e}")
            # Fallback: assume orientation is correct if we can't check
            return True
    
    def _compute_orientation_reward(self, gripper_pos, cube_pos):
        """Compute reward for good orientation (pointing toward cube)"""
        try:
            # Get end-effector orientation from PyBullet
            ee_state = p.getLinkState(self.env.robot.id, self.env.robot.eef_id)
            ee_orientation = ee_state[1]  # Quaternion [x, y, z, w]
            
            # Convert quaternion to rotation matrix
            rotation_matrix = p.getMatrixFromQuaternion(ee_orientation)
            rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
            
            # Get the z-axis of the end-effector (gripper direction)
            gripper_z_axis = rotation_matrix[:, 2]  # Third column is z-axis
            
            # Calculate direction vector from gripper to cube
            direction_to_cube = cube_pos - gripper_pos
            direction_to_cube = direction_to_cube / (np.linalg.norm(direction_to_cube) + 1e-6)  # Normalize
            
            # Calculate dot product between gripper z-axis and direction to cube
            # We want the gripper to point toward the cube (positive dot product)
            alignment = np.dot(gripper_z_axis, direction_to_cube)
            
            # Reward for good alignment (0 to 1, where 1 is perfect alignment)
            orientation_reward = max(0, alignment) * self.orientation_reward_scale
            
            return orientation_reward
            
        except:
            # Fallback: simple height-based orientation reward
            # If gripper is above cube, reward downward orientation
            height_diff = gripper_pos[2] - cube_pos[2]
            if height_diff > 0.05:  # Gripper is significantly above cube
                return 1.0  # Small reward for being above
            return 0.0
    
    def _is_approach_complete(self, obs):
        """Check if approach phase is complete - 5cm above cube with downward orientation"""
        # obs is the approach observation format: [gripper_x, gripper_y, gripper_z, cube_x, cube_y, cube_z, distance, height_diff, wrist_1, wrist_2, wrist_3]
        
        # Extract positions from the approach observation
        gripper_pos = obs[:3]      # First 3 elements are gripper position
        cube_pos = obs[3:6]        # Next 3 elements are cube position
        distance = obs[6]          # Distance is already computed
        height_diff = obs[7]       # Height difference is already computed
        
        # Check distance (5cm threshold)
        distance_ok = distance < self.approach_distance_threshold
        
        # Check height (gripper should be above cube)
        height_ok = height_diff > 0.02  # At least 2cm above cube
        
        # Check horizontal distance (should be close horizontally)
        horizontal_distance = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])
        horizontal_ok = horizontal_distance < 0.05  # Within 5cm horizontally
        
        # Check orientation (gripper should point down)
        orientation_ok = self._check_downward_orientation()
        
        # All conditions must be met for success
        return distance_ok and height_ok and horizontal_ok and orientation_ok
    
    def step(self, action):
        """Override step method for approach-specific logic"""
        # Scale actions to reasonable ranges for the robot
        scaled_action = np.array([
            action[0] * self.position_scale,   # x (scaled movement)
            action[1] * self.position_scale,   # y (scaled movement)
            action[2] * self.position_scale,   # z (scaled movement)
            action[3] * self.orientation_scale,   # roll (scaled rotation)
            action[4] * self.orientation_scale,   # pitch (scaled rotation)
            action[5] * self.orientation_scale,   # yaw (scaled rotation)
            0.085  # gripper (open - fixed)
        ])
        
        # Apply action smoothing
        if self.previous_action is not None:
            scaled_action = (self.action_smoothing_factor * self.previous_action + 
                           (1 - self.action_smoothing_factor) * scaled_action)
        self.previous_action = scaled_action.copy()
        
        # Bypass parent's action scaling and call underlying environment directly
        # This ensures the fixed orientation is preserved
        obs, reward, done, info = self.env.step(scaled_action, control_method='end')
        
        # Update episode tracking
        self.episode_length += 1
        
        # Convert observation to gym format (same as parent environment)
        gym_obs = super()._get_gym_observation(obs)
        
        # Get approach-specific observation
        approach_obs = self._get_approach_observation(gym_obs)
        
        # Get gripper and cube positions for distance calculations
        # gym_obs is numpy array: [joint_positions(6), ee_pos(3), cube_pos(3), target_pos(3), distance(1)]
        joint_positions = gym_obs[:6]
        ee_pos = gym_obs[6:9]          # End-effector position
        cube_pos = gym_obs[9:12]       # Cube position
        
        # Use the actual end-effector position from the observation
        gripper_pos = ee_pos
        
        # Check for collisions
        collision_info = self._check_collisions(gripper_pos)
        floor_collision = collision_info['floor_collision']
        self_collision = collision_info['self_collision']
        body_collision = collision_info['body_collision']
        any_collision = collision_info['any_collision']
        

        
        # Check if gripper is too far away from cube (early termination)
        distance_to_cube = np.linalg.norm(gripper_pos - cube_pos)
        too_far_away = distance_to_cube > self.too_far_threshold
        
        # Check if approach is complete
        approach_complete = self._is_approach_complete(gym_obs)
        
        # Compute approach-specific reward
        approach_reward = self._compute_approach_reward(gym_obs, action)
        
        # Get current progressive penalty (will be passed from train.py)
        # For now, use a default value that will be updated by train.py
        current_collision_penalty = self._get_progressive_collision_penalty()
        
        # Add penalties with progressive system
        if too_far_away:
            approach_reward -= 5.0  # Penalty for going too far away
        
        if floor_collision:
            approach_reward += current_collision_penalty  # Progressive penalty for gripper floor collision
        
        if self_collision:
            approach_reward += current_collision_penalty * 2  # Progressive penalty for self-collision
        
        if body_collision:
            approach_reward += current_collision_penalty * 2  # Progressive penalty for body collision
        
        # Add reward for proper approach pose
        if self._check_proper_approach_pose(gripper_pos, cube_pos):
            approach_reward += 3.0  # Bonus for being in proper approach pose
        
        # Episode termination - don't terminate on collision, let agent learn
        terminated = approach_complete or too_far_away or (self.episode_length >= self.max_episode_length)
        truncated = (self.episode_length >= self.max_episode_length)
        
        # Update info (positions already extracted above)
        
        info.update({
            'phase': 'approach',
            'approach_complete': approach_complete,
            'too_far_away': too_far_away,
            'floor_collision': floor_collision,
            'self_collision': self_collision,
            'body_collision': body_collision,
            'any_collision': any_collision,
            'proper_pose': self._check_proper_approach_pose(gripper_pos, cube_pos),
            'distance_to_cube': np.linalg.norm(gripper_pos - cube_pos),
            'gripper_position': gripper_pos,
            'episode_length': self.episode_length,
            'cube_distance': distance_to_cube,  # Distance to cube (already calculated above)
            'is_success': approach_complete,
            'termination_reason': 'success' if approach_complete else 'collision' if any_collision else 'too_far' if too_far_away else 'timeout'
        })
        return approach_obs, approach_reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset with approach-specific initialization"""
        obs, info = super().reset(seed, options)
        
        # Reset action smoothing
        self.previous_action = None
        

        
        # Convert to approach observation format
        approach_obs = self._get_approach_observation(obs)
        
        return approach_obs, info
    
    def set_training_step(self, step):
        """Set the current training step for progressive penalties"""
        self.current_training_steps = step
    
    def set_total_training_steps(self, total_steps):
        """Set total training steps from train.py"""
        self.total_training_steps = total_steps
        print(f"🎯 Progressive penalty system: {total_steps:,} total steps")
    
    def set_parameters(self, position_scale=None, orientation_scale=None, 
                      approach_threshold=None, too_far_threshold=None,
                      min_gripper_height=None, min_approach_height=None,
                      collision_penalty=None, self_collision_penalty=None, body_collision_penalty=None,
                      orientation_reward_scale=None, enable_orientation_reward=None):
        """Dynamically adjust environment parameters"""
        if position_scale is not None:
            self.position_scale = position_scale
        if orientation_scale is not None:
            self.orientation_scale = orientation_scale
        if approach_threshold is not None:
            self.approach_distance_threshold = approach_threshold
        if too_far_threshold is not None:
            self.too_far_threshold = too_far_threshold
        if min_gripper_height is not None:
            self.min_gripper_height = min_gripper_height
        if min_approach_height is not None:
            self.min_approach_height = min_approach_height
        if collision_penalty is not None:
            self.collision_penalty = collision_penalty
        if self_collision_penalty is not None:
            self.self_collision_penalty = self_collision_penalty
        if body_collision_penalty is not None:
            self.body_collision_penalty = body_collision_penalty
        if orientation_reward_scale is not None:
            self.orientation_reward_scale = orientation_reward_scale
        if enable_orientation_reward is not None:
            self.enable_orientation_reward = enable_orientation_reward
        
        print(f"🔧 Parameters updated:")
        print(f"   Position scale: {self.position_scale}m per step")
        print(f"   Orientation scale: {self.orientation_scale}rad per step")
        print(f"   Approach threshold: {self.approach_distance_threshold}m")
        print(f"   Too far threshold: {self.too_far_threshold}m")
        print(f"   Min gripper height: {self.min_gripper_height}m")
        print(f"   Collision penalty: {self.collision_penalty}")
        print(f"   Self-collision penalty: {self.self_collision_penalty}")
        print(f"   Orientation reward scale: {self.orientation_reward_scale}")
        print(f"   Enable orientation reward: {self.enable_orientation_reward}")