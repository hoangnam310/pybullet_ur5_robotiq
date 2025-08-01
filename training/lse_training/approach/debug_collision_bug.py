#!/usr/bin/env python3
"""
Debug script to test collision detection when gripper touches floor
"""
import sys
import os
import numpy as np
import pybullet as p

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env import ApproachEnv

def test_gripper_floor_collision():
    """Test what happens when gripper touches floor"""
    print("🔍 Testing Gripper Floor Collision Detection")
    
    # Create environment
    env = ApproachEnv(render_mode=None)
    
    # Reset environment
    obs, _ = env.reset()
    
    print("\n📋 Initial state:")
    ee_pos = obs[6:9]  # End-effector position
    print(f"   End-effector position: {ee_pos}")
    
    # Get initial collision state
    print("\n🔍 Initial collision check:")
    collisions = env._check_collisions(ee_pos)
    print(f"   Floor collision: {collisions['floor_collision']}")
    print(f"   Body collision: {collisions['body_collision']}")
    print(f"   Self collision: {collisions['self_collision']}")
    
    # Get gripper links
    gripper_links = env._get_gripper_link_ids()
    print(f"   Gripper link IDs: {gripper_links}")
    
    # Test with action that moves gripper down to touch floor
    print("\n🎯 Testing with aggressive downward action:")
    
    # Create action to move gripper down aggressively
    action = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])  # Move down in Z
    
    # Take many steps to force gripper to touch floor
    for step in range(50):  # More steps to ensure contact
        obs, reward, terminated, truncated, info = env.step(action)
        ee_pos = obs[6:9]
        
        print(f"   Step {step+1}: EE pos = {ee_pos}, height = {ee_pos[2]:.4f}")
        
        # Check collisions
        collisions = env._check_collisions(ee_pos)
        print(f"     Floor collision: {collisions['floor_collision']}")
        print(f"     Body collision: {collisions['body_collision']}")
        print(f"     Self collision: {collisions['self_collision']}")
        
        # Check if gripper is very close to floor
        if ee_pos[2] < 0.02:  # Very close to floor
            print(f"     ⚠️  Gripper very close to floor! Height: {ee_pos[2]:.4f}")
            
            # Run detailed debug at this point
            print(f"     🔍 Running detailed collision debug...")
            debug_info = env.debug_collision_detection()
            
            if debug_info and len(debug_info['floor_contacts']) > 0:
                print("     Floor contact details:")
                for i, contact in enumerate(debug_info['floor_contacts']):
                    link_a = contact[3]  # Robot link
                    contact_pos = contact[5]  # Contact position
                    contact_depth = contact[8]  # Penetration depth
                    print(f"       Contact {i}: Link {link_a} at pos {contact_pos}, depth {contact_depth:.4f}")
                    
                    # Check if this is a gripper link
                    if link_a in gripper_links:
                        print(f"       ✅ This is a GRIPPER link!")
                    else:
                        print(f"       ❌ This is a BODY link!")
        
        if terminated:
            print(f"     Episode terminated: {info.get('termination_reason', 'unknown')}")
            break
    
    # Final detailed debug
    print("\n🔍 Final detailed collision debug:")
    debug_info = env.debug_collision_detection()
    
    if debug_info:
        print(f"   Floor contacts: {len(debug_info['floor_contacts'])}")
        print(f"   Self contacts: {len(debug_info['self_contacts'])}")
        
        if len(debug_info['floor_contacts']) > 0:
            print("   Floor contact details:")
            for i, contact in enumerate(debug_info['floor_contacts']):
                link_a = contact[3]  # Robot link
                contact_pos = contact[5]  # Contact position
                contact_depth = contact[8]  # Penetration depth
                print(f"     Contact {i}: Link {link_a} at pos {contact_pos}, depth {contact_depth:.4f}")
                
                # Check if this is a gripper link
                if link_a in gripper_links:
                    print(f"     ✅ This is a GRIPPER link!")
                else:
                    print(f"     ❌ This is a BODY link!")
    
    print("\n✅ Gripper floor collision test completed!")

if __name__ == "__main__":
    test_gripper_floor_collision() 