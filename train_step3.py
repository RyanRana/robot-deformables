#!/usr/bin/env python3
"""
Step 3: Model Training - Imitation Learning Pretraining

Trains DiffusionPolicy on streaming data from Step 1.
"""

import torch
import numpy as np
import gymnasium as gym
from data_preparation import merged_generator
import deformable_handover_env
from diffusion_policy import DiffusionPolicy, DiffusionPolicyTrainer
import tempfile
import os


def train_il_policy(
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    validation_batches: int = 20,
    target_mse: float = 0.15,
    device: str = 'auto'
):
    """
    Train Imitation Learning policy.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        validation_batches: Number of validation batches
        target_mse: Target validation MSE
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
    
    Returns:
        Trained policy and final validation MSE
    """
    
    print("=" * 70)
    print(" STEP 3: IMITATION LEARNING PRETRAINING")
    print("=" * 70)
    print()
    
    # Create environment to get spaces
    print("Creating environment...")
    env = gym.make('DeformableHandover-v0')
    
    # Initialize policy
    print(f"Initializing DiffusionPolicy...")
    policy = DiffusionPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=256
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    
    # Initialize trainer
    trainer = DiffusionPolicyTrainer(policy, device=device, lr=lr)
    print()
    
    # Training loop
    print(f"Training for {num_epochs} epochs...")
    print("-" * 70)
    
    best_val_mse = float('inf')
    epoch_losses = []
    
    for epoch in range(num_epochs):
        # Create new generator for this epoch
        train_gen = merged_generator(batch_size=batch_size)
        
        epoch_loss = 0
        num_batches = 0
        
        # Train on all available batches
        for batch_idx, batch in enumerate(train_gen):
            loss = trainer.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            
            # Limit batches per epoch for reasonable training time
            if num_batches >= 100:  # Process 100 batches per epoch
                break
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_loss)
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            val_mse, val_mae = validate_policy(trainer, validation_batches, batch_size)
            
            print(f"Epoch {epoch + 1:3d}/{num_epochs}: "
                  f"Train Loss={avg_loss:.4f}, "
                  f"Val MSE={val_mse:.4f}, "
                  f"Val MAE={val_mae:.4f}")
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
        else:
            print(f"Epoch {epoch + 1:3d}/{num_epochs}: Train Loss={avg_loss:.4f}")
    
    print("-" * 70)
    print()
    
    # Final validation
    print("Final validation on 20 holdout batches...")
    val_mse, val_mae = validate_policy(trainer, validation_batches, batch_size)
    
    print(f"  Validation MSE: {val_mse:.4f}")
    print(f"  Validation MAE: {val_mae:.4f}")
    print()
    
    # Decision criterion check
    print("=" * 70)
    print("DECISION CRITERION CHECK")
    print("=" * 70)
    
    if val_mse > 0.2:
        print(f"⚠ Validation MSE ({val_mse:.4f}) > 0.2")
        print("  Recommendation: Increase epochs to 150 and retrain")
        print("  Status: NEEDS MORE TRAINING")
        criterion_met = False
    elif val_mse > target_mse:
        print(f"⚠ Validation MSE ({val_mse:.4f}) > {target_mse}")
        print("  Recommendation: Continue training or adjust hyperparameters")
        print("  Status: MARGINAL")
        criterion_met = False
    else:
        print(f"✓ Validation MSE ({val_mse:.4f}) <= {target_mse}")
        print("  Status: PASSED")
        criterion_met = True
    
    print("=" * 70)
    print()
    
    # Save checkpoint temporarily
    temp_dir = tempfile.gettempdir()
    checkpoint_path = os.path.join(temp_dir, 'diffusion_policy_checkpoint.pt')
    
    print(f"Saving checkpoint to: {checkpoint_path}")
    trainer.save_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint saved (kept for Step 4)")
    print()
    
    # Cleanup
    env.close()
    
    return policy, trainer, val_mse, criterion_met, checkpoint_path


def validate_policy(
    trainer: DiffusionPolicyTrainer,
    num_batches: int,
    batch_size: int
) -> tuple:
    """
    Validate policy on holdout data.
    
    Args:
        trainer: Policy trainer
        num_batches: Number of validation batches
        batch_size: Batch size
    
    Returns:
        Average MSE and MAE
    """
    val_gen = merged_generator(batch_size=batch_size)
    
    total_mse = 0
    total_mae = 0
    
    for batch_idx, batch in enumerate(val_gen):
        if batch_idx >= num_batches:
            break
        
        mse, mae = trainer.validate(batch)
        total_mse += mse
        total_mae += mae
    
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_mse, avg_mae


if __name__ == "__main__":
    # Train with default parameters (optimized for cloud GPU)
    policy, trainer, val_mse, criterion_met, checkpoint_path = train_il_policy(
        num_epochs=100,
        batch_size=64,  # Increased for RTX 5060 Ti (15.9 GB VRAM)
        lr=1e-4,
        validation_batches=20,
        target_mse=0.15,
        device='cuda'  # Force CUDA for cloud GPU
    )
    
    # Keep checkpoint for Step 4
    print(f"\n💾 IL checkpoint saved at: {checkpoint_path}")
    print("   (This will be used by Step 4 RL fine-tuning)")
    
    print("\n" + "=" * 70)
    if criterion_met:
        print("✅ STEP 3 COMPLETE - Policy meets validation criteria!")
    else:
        print("⚠ STEP 3 NEEDS ADJUSTMENT - Review recommendations above")
    print("=" * 70)

