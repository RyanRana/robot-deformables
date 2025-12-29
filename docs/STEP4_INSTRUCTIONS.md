# Step 4: RL Fine-Tuning - Quick Start

## Prerequisites

Make sure Step 3 (IL pretraining) completed successfully and saved a checkpoint.

## Installation

If you haven't already, install stable-baselines3:

```bash
pip install stable-baselines3
```

## Run Step 4

### On Vast.ai (via JupyterLab or SSH):

```bash
cd /workspace
python3 train_step4_rl.py
```

### What it does:

1. **Loads IL policy** from Step 3 checkpoint
2. **Creates hybrid IL+RL policy** with Transformer fusion
3. **Trains with PPO** for 50,000 timesteps (reduced for testing)
4. **Uses curriculum learning**:
   - Phase 1: 10,000 steps with rigid sims (IL weight: 0.7)
   - Phase 2: 40,000 steps with deformable (IL weight: 0.3, RL weight: 0.7)
5. **Evaluates** on 100 episodes
6. **Checks success rate** ≥ 85%

## Key Features

✅ **Multi-modal Transformer fusion** (image + effort + IMU + audio)  
✅ **Hybrid IL+RL blending** with adaptive weights  
✅ **Safety constraints** (scales action if effort > 50N)  
✅ **Curriculum learning** (rigid → deformable)  
✅ **Custom PPO** (works without stable-baselines3)

## Expected Output

```
======================================================================
 STEP 4: RL FINE-TUNING WITH PPO
======================================================================

Creating environment...
Initializing hybrid IL+RL policy...
  Hybrid policy parameters: 2,345,678

Starting RL training with curriculum learning...
----------------------------------------------------------------------
Phase 1: Curriculum (rigid sims) - 10000 steps
Step 0/10000: Avg Reward=-8.50
Step 1000/10000: Avg Reward=-7.20
...

Phase 2: Full deformable - 40000 steps
Step 10000/50000: Avg Reward=-6.50
...

Evaluating trained policy...
  Evaluated 20/100 episodes...
  Evaluated 40/100 episodes...
  ...

Evaluation Results:
  Success rate: 87.0%

✓ Success rate (87.0%) >= 85%
  Status: PASSED

✅ STEP 4 COMPLETE - Policy meets success criteria!
======================================================================
```

## Training Time

- **With GPU**: ~2-3 hours for 50,000 timesteps
- **Full training** (500,000 timesteps): ~20-24 hours

## If Step 3 Didn't Complete

The script will warn you and train from scratch (not recommended). To fix:
1. Go back and complete Step 3 first
2. Or provide a checkpoint path manually

## Decision Criteria

- **Success rate ≥ 85%**: ✅ Pass
- **75% ≤ Success < 85%**: ⚠ Marginal (consider adversarial training)
- **Success < 75%**: ❌ Need adversarial perturbations + retrain

## Adjusting Parameters

In `train_step4_rl.py`, you can modify:

```python
train_rl_finetuning(
    total_timesteps=500000,  # Increase for longer training
    curriculum_timesteps=100000,  # More curriculum steps
    device='cuda'
)
```

## Next Steps

Once Step 4 passes:
- Download the final model
- Deploy for testing
- Or continue to Step 5 (if applicable)

