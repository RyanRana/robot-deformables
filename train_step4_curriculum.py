#!/usr/bin/env python3
"""
Step 4: RL Fine-Tuning with Curriculum Learning

Trains in 3 stages with increasing difficulty:
- Stage 1: Human very close (0.25m) - Easy handover
- Stage 2: Human medium distance (0.45m)  
- Stage 3: Human far distance (0.7m+) - Full task
"""

import torch
import gymnasium as gym
import os
import tempfile
import numpy as np

from diffusion_policy import DiffusionPolicy
from train_step4_rl import HybridILRLPolicy, CustomPPO, evaluate_policy


def train_curriculum_stage(
    stage_name: str,
    hybrid_policy,
    env,
    ppo,
    total_timesteps: int,
    il_weight_start: float,
    il_weight_end: float,
    device='cuda'
):
    """Train one curriculum stage."""
    
    print("\n" + "=" * 70)
    print(f" {stage_name}")
    print("=" * 70)
    print(f"  Timesteps: {total_timesteps}")
    print(f"  IL weight: {il_weight_start:.2f} → {il_weight_end:.2f}")
    print()
    
    total_rollouts = total_timesteps // 2048
    
    for rollout_idx in range(total_rollouts):
        # Compute current IL weight
        progress = rollout_idx / total_rollouts
        il_weight = il_weight_start - ((il_weight_start - il_weight_end) * progress)
        rl_weight = 1.0 - il_weight
        hybrid_policy.set_blend_weights(il_weight, rl_weight)
        
        # Train for one rollout
        current_steps = rollout_idx * 2048
        if rollout_idx % 5 == 0:
            print(f"  Rollout {rollout_idx}/{total_rollouts} (Step {current_steps}): IL={il_weight:.2f}, RL={rl_weight:.2f}")
        
        ppo.learn(total_timesteps=2048, log_interval=2)
    
    print(f"\n✓ {stage_name} complete!")
    print("=" * 70)


def train_with_curriculum(device='cuda'):
    """
    Main curriculum training function.
    """
    
    print("=" * 70)
    print(" STEP 4: CURRICULUM LEARNING - 3 STAGES")
    print("=" * 70)
    print()
    print("Strategy: Start easy, gradually increase difficulty")
    print("  Stage 1: Human at 0.25m (very close)")
    print("  Stage 2: Human at 0.45m (medium)")
    print("  Stage 3: Human at 0.7m+ (full distance)")
    print()
    
    # Load IL checkpoint
    il_checkpoint = os.path.join(tempfile.gettempdir(), 'diffusion_policy_checkpoint.pt')
    
    if not os.path.exists(il_checkpoint):
        print("❌ Error: No IL checkpoint found from Step 3.")
        print("   Please run train_step3.py first!")
        return
    
    # Stage 1: Very close human (easy)
    print("\n" + "=" * 70)
    print(" STAGE 1: EASY (Human at 0.25m)")
    print("=" * 70)
    
    env_stage1 = gym.make('DeformableHandover-v0', human_distance=0.25)
    il_policy = DiffusionPolicy(env_stage1.observation_space, env_stage1.action_space, hidden_dim=256)
    
    print(f"Loading IL checkpoint from: {il_checkpoint}")
    checkpoint = torch.load(il_checkpoint, map_location=device)
    il_policy.load_state_dict(checkpoint['policy_state_dict'])
    
    hybrid_policy = HybridILRLPolicy(il_policy, action_dim=6, hidden_dim=256)
    hybrid_policy = hybrid_policy.to(device)
    
    ppo_stage1 = CustomPPO(hybrid_policy, env_stage1, lr=1e-4, device=device)
    
    train_curriculum_stage(
        "STAGE 1: Easy Handover",
        hybrid_policy,
        env_stage1,
        ppo_stage1,
        total_timesteps=100000,  # 100k for stage 1
        il_weight_start=0.95,
        il_weight_end=0.70,
        device=device
    )
    
    # Evaluate stage 1
    print("\nEvaluating Stage 1...")
    success_rate_1 = evaluate_policy(hybrid_policy, env_stage1, n_episodes=50, device=device)
    print(f"  Stage 1 Success Rate: {success_rate_1:.1f}%")
    env_stage1.close()
    
    if success_rate_1 < 50:
        print("⚠ Warning: Stage 1 success rate < 50%. Consider retraining with more steps.")
    
    # Stage 2: Medium distance
    print("\n" + "=" * 70)
    print(" STAGE 2: MEDIUM (Human at 0.45m)")
    print("=" * 70)
    
    env_stage2 = gym.make('DeformableHandover-v0', human_distance=0.45)
    ppo_stage2 = CustomPPO(hybrid_policy, env_stage2, lr=8e-5, device=device)  # Lower LR
    
    train_curriculum_stage(
        "STAGE 2: Medium Handover",
        hybrid_policy,
        env_stage2,
        ppo_stage2,
        total_timesteps=150000,  # 150k for stage 2
        il_weight_start=0.70,
        il_weight_end=0.45,
        device=device
    )
    
    # Evaluate stage 2
    print("\nEvaluating Stage 2...")
    success_rate_2 = evaluate_policy(hybrid_policy, env_stage2, n_episodes=50, device=device)
    print(f"  Stage 2 Success Rate: {success_rate_2:.1f}%")
    env_stage2.close()
    
    # Stage 3: Full distance
    print("\n" + "=" * 70)
    print(" STAGE 3: HARD (Human at 0.7m+)")
    print("=" * 70)
    
    env_stage3 = gym.make('DeformableHandover-v0')  # Default distance
    ppo_stage3 = CustomPPO(hybrid_policy, env_stage3, lr=5e-5, device=device)  # Even lower LR
    
    train_curriculum_stage(
        "STAGE 3: Full Distance Handover",
        hybrid_policy,
        env_stage3,
        ppo_stage3,
        total_timesteps=200000,  # 200k for stage 3
        il_weight_start=0.45,
        il_weight_end=0.30,
        device=device
    )
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(" FINAL EVALUATION")
    print("=" * 70)
    
    success_rate_final = evaluate_policy(hybrid_policy, env_stage3, n_episodes=100, device=device)
    
    print(f"\nFinal Results:")
    print(f"  Stage 1 (0.25m): {success_rate_1:.1f}%")
    print(f"  Stage 2 (0.45m): {success_rate_2:.1f}%")
    print(f"  Stage 3 (0.7m+): {success_rate_final:.1f}%")
    print()
    
    # Save model
    save_path = os.path.join(tempfile.gettempdir(), 'hybrid_rl_curriculum.pt')
    torch.save({
        'policy_state_dict': hybrid_policy.state_dict(),
        'stage1_success': success_rate_1,
        'stage2_success': success_rate_2,
        'final_success': success_rate_final,
    }, save_path)
    print(f"Model saved to: {save_path}")
    print()
    
    # Decision criterion
    print("=" * 70)
    print("DECISION CRITERION CHECK")
    print("=" * 70)
    
    if success_rate_final >= 85:
        print(f"✓ Final success rate ({success_rate_final:.1f}%) >= 85%")
        print("  Status: PASSED")
        criterion_met = True
    elif success_rate_final >= 75:
        print(f"⚠ Final success rate ({success_rate_final:.1f}%) >= 75% but < 85%")
        print("  Status: MARGINAL - Close to target!")
        criterion_met = False
    else:
        print(f"⚠ Final success rate ({success_rate_final:.1f}%) < 75%")
        print("  Recommendation: Increase Stage 3 timesteps or adjust rewards")
        criterion_met = False
    
    print("=" * 70)
    print()
    
    env_stage3.close()
    
    return hybrid_policy, success_rate_final, criterion_met


if __name__ == "__main__":
    print("\n🎓 CURRICULUM LEARNING APPROACH")
    print("=" * 70)
    print("Total training: ~450k timesteps (~10-12 hours)")
    print("Cost estimate: ~$0.75-$0.90")
    print()
    print("This approach:")
    print("  1. Learns easy handovers first (confidence building)")
    print("  2. Gradually increases difficulty")
    print("  3. Transfers learned skills across stages")
    print("  4. Much more sample efficient than random training!")
    print("=" * 70)
    print()
    
    policy, success_rate, criterion_met = train_with_curriculum(device='cuda')
    
    print("\n" + "=" * 70)
    if criterion_met:
        print("✅ CURRICULUM TRAINING COMPLETE - Policy meets success criteria!")
    else:
        print("⚠ CURRICULUM TRAINING COMPLETE - Review results above")
    print("=" * 70)

