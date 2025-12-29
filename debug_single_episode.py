#!/usr/bin/env python3
"""
Debug script to run a single episode and print detailed diagnostics.
"""

import torch
import numpy as np
import gymnasium as gym
import os
import tempfile

from diffusion_policy import DiffusionPolicy
from train_step4_rl import HybridILRLPolicy

def debug_episode(use_il_checkpoint=True, device='cuda'):
    """Run a single episode with detailed logging."""
    
    print("=" * 70)
    print(" DEBUG: Single Episode Diagnostics")
    print("=" * 70)
    print()
    
    # Create environment
    print("Creating environment...")
    env = gym.make('DeformableHandover-v0', adversarial_mode=False)
    
    # Load policy
    print("Loading policy...")
    il_policy = DiffusionPolicy(env.observation_space, env.action_space, hidden_dim=256)
    
    if use_il_checkpoint:
        checkpoint_path = os.path.join(tempfile.gettempdir(), 'diffusion_policy_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading IL checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            il_policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            print("Warning: No IL checkpoint found, using random policy")
    
    # Create hybrid policy
    hybrid_policy = HybridILRLPolicy(il_policy, action_dim=6, hidden_dim=256)
    hybrid_policy = hybrid_policy.to(device)
    hybrid_policy.set_blend_weights(0.9, 0.1)  # Mostly IL for debugging
    
    # Run episode
    print("\nRunning episode...")
    print("-" * 70)
    
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(100):
        # Prepare observation
        obs_tensor = {}
        for key in ['image', 'effort', 'imu', 'audio']:
            if key == 'image':
                tensor = torch.from_numpy(obs[key]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            elif key == 'effort':
                tensor = torch.from_numpy(obs[key]).unsqueeze(0).float() / 50.0
            elif key == 'imu':
                tensor = torch.from_numpy(obs[key]).unsqueeze(0).float() / 5.0
            else:
                tensor = torch.from_numpy(obs[key]).unsqueeze(0).float()
            obs_tensor[key] = tensor.to(device)
        
        # Get action
        with torch.no_grad():
            action = hybrid_policy(obs_tensor, return_value=False, deterministic=True)
            action_np = action.squeeze().cpu().numpy()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        
        episode_reward += reward
        
        # Print detailed step info
        print(f"Step {step:3d}: reward={reward:6.3f}, "
              f"effort={np.linalg.norm(obs['effort']):5.1f}N, "
              f"towel_z={info.get('towel_z', 0):.3f}m, "
              f"grasped={info.get('grasped', False)}, "
              f"contact={info.get('contact_force', 0):.2f}N")
        
        if step % 10 == 0:
            print(f"  Action: {action_np}")
            print(f"  Distance to human: {info.get('arm_to_human_dist', 0):.3f}m")
        
        obs = next_obs
        
        if terminated or truncated:
            print("-" * 70)
            print(f"Episode ended at step {step + 1}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            print(f"  Success: {info.get('handover_achieved', False)}")
            print(f"  Failure reason: {info.get('failure_reason', 'none')}")
            print(f"  Total reward: {episode_reward:.2f}")
            break
    
    if not (terminated or truncated):
        print("-" * 70)
        print(f"Episode completed all 100 steps")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Success: {info.get('handover_achieved', False)}")
    
    print("=" * 70)
    print()
    
    env.close()
    
    return episode_reward, info


if __name__ == "__main__":
    # Run debug episode
    reward, info = debug_episode(use_il_checkpoint=True, device='cuda')
    
    print("\nSummary:")
    print(f"  Final reward: {reward:.2f}")
    print(f"  Success: {info.get('handover_achieved', False)}")
    
    if reward < -5:
        print("\n⚠️  Reward is very negative - likely issues:")
        print("  1. Check if IL policy is loaded correctly")
        print("  2. Verify action scaling matches environment expectations")
        print("  3. Check if observations are normalized properly")
    elif reward > 0:
        print("\n✅ Positive reward - policy is making progress!")
    else:
        print("\n⚠️  Neutral/slightly negative - marginal performance")

