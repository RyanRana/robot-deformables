#!/usr/bin/env python3
"""
Integration test: Steps 1 & 2 working together.

Demonstrates the complete pipeline from data loading to environment interaction.
"""

import gymnasium as gym
import numpy as np
# Import environment module to register it
import deformable_handover_env
from data_preparation import merged_generator


def test_integration():
    """Test integration of Step 1 and Step 2."""
    
    print("\n" + "=" * 70)
    print(" INTEGRATION TEST: STEPS 1 & 2")
    print("=" * 70 + "\n")
    
    # Part 1: Verify Step 1 data generator
    print("Part 1: Verifying Step 1 data generator...")
    print("-" * 70)
    
    try:
        generator = merged_generator(batch_size=32)
        batch = next(generator)
        
        print(f"✓ Generator created and batch loaded")
        print(f"  Batch size: {batch['action'].shape[0]}")
        print(f"  Image shape: {batch['observation']['image'].shape}")
        print(f"  Action shape: {batch['action'].shape}")
        print("  Step 1: WORKING ✓\n")
        step1_ok = True
    except Exception as e:
        print(f"✗ Step 1 failed: {e}\n")
        step1_ok = False
        return False
    
    # Part 2: Verify Step 2 environment
    print("Part 2: Verifying Step 2 environment...")
    print("-" * 70)
    
    try:
        env = gym.make('DeformableHandover-v0')
        
        print(f"✓ Environment created: {env.spec.id}")
        print(f"  Observation space: Dict with {len(env.observation_space.spaces)} keys")
        print(f"  Action space: {env.action_space.shape}")
        print("  Step 2: WORKING ✓\n")
        step2_ok = True
    except Exception as e:
        print(f"✗ Step 2 failed: {e}\n")
        step2_ok = False
        return False
    
    # Part 3: Verify integration
    print("Part 3: Verifying Steps 1 & 2 integration...")
    print("-" * 70)
    
    try:
        # Reset environment (uses Step 1 generator internally)
        obs, info = env.reset(seed=42)
        
        print(f"✓ Environment reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Image: {obs['image'].shape}, dtype={obs['image'].dtype}")
        print(f"  Effort: {obs['effort'].shape}, dtype={obs['effort'].dtype}")
        print(f"  IMU: {obs['imu'].shape}, dtype={obs['imu'].dtype}")
        print(f"  Audio: {obs['audio'].shape}, dtype={obs['audio'].dtype}")
        
        # Take a step
        action = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
        next_obs, reward, term, trunc, info = env.step(action)
        
        print(f"\n✓ Environment step successful")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {term}, Truncated: {trunc}")
        print(f"  Towel height: {info['towel_z']:.3f}m")
        print(f"  Contact force: {info['contact_force']:.2f}N")
        
        print("\n  Integration: WORKING ✓\n")
        integration_ok = True
    except Exception as e:
        print(f"\n✗ Integration failed: {e}\n")
        integration_ok = False
        return False
    
    # Part 4: Run a complete episode
    print("Part 4: Running complete episode...")
    print("-" * 70)
    
    try:
        obs, _ = env.reset(seed=100)
        episode_reward = 0
        steps = 0
        
        for step in range(50):
            # Simple policy: move towards human slowly
            action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
            obs, reward, term, trunc, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if step % 10 == 0:
                print(f"  Step {step:2d}: reward={reward:6.2f}, z={info['towel_z']:.3f}m, force={info['contact_force']:.2f}N")
            
            if term or trunc:
                break
        
        print(f"\n✓ Episode completed")
        print(f"  Total steps: {steps}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Handover achieved: {info.get('handover_achieved', False)}")
        
        print("\n  Episode execution: WORKING ✓\n")
        episode_ok = True
    except Exception as e:
        print(f"\n✗ Episode failed: {e}\n")
        episode_ok = False
        return False
    finally:
        env.close()
    
    # Final summary
    print("=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Step 1 (Data Pipeline): {'PASS' if step1_ok else 'FAIL'}")
    print(f"✓ Step 2 (Environment): {'PASS' if step2_ok else 'FAIL'}")
    print(f"✓ Integration: {'PASS' if integration_ok else 'FAIL'}")
    print(f"✓ Episode Execution: {'PASS' if episode_ok else 'FAIL'}")
    
    all_ok = step1_ok and step2_ok and integration_ok and episode_ok
    
    if all_ok:
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        print("Steps 1 & 2 are working together seamlessly!")
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("Review the output above for details.")
    
    print("=" * 70 + "\n")
    
    return all_ok


if __name__ == "__main__":
    success = test_integration()
    
    if success:
        print("NEXT STEPS:")
        print("  1. Design policy network (Step 3)")
        print("  2. Implement training algorithm (PPO/SAC)")
        print("  3. Train agent on the environment")
        print("  4. Evaluate and deploy\n")
    
    exit(0 if success else 1)

