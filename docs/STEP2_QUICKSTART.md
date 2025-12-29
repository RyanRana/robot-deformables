# Quick Start Guide - Step 2 Environment

## Installation

Already done in Step 1! The environment uses Gymnasium which is already installed.

## Verify Installation

```bash
python3 verify_step2.py
```

Expected output:
```
✅ ALL VERIFICATIONS PASSED
Environment is ready for Step 3 (Policy Training)
```

## Basic Usage

### Option 1: Direct Use

```python
import gymnasium as gym

# Create environment
env = gym.make('DeformableHandover-v0')

# Reset
obs, info = env.reset(seed=42)
# obs is a dict with keys: 'image', 'effort', 'imu', 'audio'

# Take step
action = env.action_space.sample()  # Random action
next_obs, reward, terminated, truncated, info = env.step(action)

# Cleanup
env.close()
```

### Option 2: Episode Loop

```python
import gymnasium as gym

env = gym.make('DeformableHandover-v0')

for episode in range(10):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(100):
        # Your policy here (or random for testing)
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: Reward={episode_reward:.1f}")

env.close()
```

### Option 3: Access Observations

```python
import gymnasium as gym

env = gym.make('DeformableHandover-v0')
obs, _ = env.reset()

# Access individual observations
image = obs['image']          # (84, 84, 3) uint8 - RGB image
effort = obs['effort']        # (6,) float32 - Joint efforts
imu = obs['imu']              # (6,) float32 - IMU data
audio = obs['audio']          # (16000,) float32 - 1s audio at 16kHz

print(f"Image shape: {image.shape}")
print(f"Effort: {effort}")
print(f"Audio duration: {len(audio)/16000}s")
```

## Environment Details

### Observation Space

```python
Dict({
    'image': Box(0, 255, (84, 84, 3), uint8),
    'effort': Box(-100, 100, (6,), float32),
    'imu': Box(-10, 10, (6,), float32),
    'audio': Box(0, 1, (16000,), float32)
})
```

### Action Space

```python
Box(-1, 1, (6,), float32)
# 6DoF end-effector control (normalized)
```

### Reward Structure

- **+10**: Successful handover (effort change > 5N while grasped)
- **-1**: Per step (time penalty)
- **-50**: Drop object (z < 0.1m)
- **-50**: Excessive force (> 50N)

### Episode Termination

**Terminated** (task complete/failed):
- Successful handover detected
- Object dropped (z < 0.1m)
- Excessive force applied (> 50N)

**Truncated** (max steps):
- 100 steps reached

## Testing

### Run Full Test Suite

```bash
python3 test_environment.py
```

This runs:
- 10 episodes with random actions
- Shape verification
- Type verification
- Decision criterion check

### Run Quick Verification

```bash
python3 verify_step2.py
```

Faster verification of key functionality.

## Integration with Step 1

The environment automatically pulls observations from the Step 1 data pipeline:

```python
# Step 1 generator is called internally
# No need to manually manage data loading
env = gym.make('DeformableHandover-v0')
obs, _ = env.reset()  # Automatically pulls from merged_generator()
```

## Physics Simulation

### Current Setup
- **Simulator**: Simplified physics (PyBullet unavailable)
- **Timestep**: 1/240s (240 Hz)
- **Control frequency**: 30 Hz
- **Gravity**: -9.81 m/s² (scaled 0.5x for stability)

### Towel Properties
- **Mass**: 0.2 kg
- **Stiffness**: 200 (spring constant)
- **Initial position**: [0, 0, 0.5]m (grasped)
- **Damping**: 0.8 (velocity damping)

### Human Avatar
- **Position**: [0.5, 0, 0.5]m ± 0.1m (random)
- **Type**: Kinematic body
- **Interaction range**: 0.2m
- **Pull force**: Up to 8N (distance-dependent)

## Troubleshooting

### Issue: Low average reward
**Symptom**: Average reward << -10
**Solution**: Physics already tuned. This is expected with random policy. Train a policy for better performance!

### Issue: Object drops immediately
**Symptom**: z < 0.1m in first few steps
**Solution**: Already fixed with increased stiffness (200) and initial grasp state.

### Issue: Data loading slow
**Symptom**: First reset takes 2-3 minutes
**Solution**: This is one-time loading from Step 1. Subsequent resets are fast (<100ms).

## Performance

| Operation | Time |
|-----------|------|
| First reset | ~2-3 min (one-time data loading) |
| Subsequent resets | ~100ms |
| Step | ~1ms |
| Episode (100 steps) | ~0.2s |

## Next Steps

Your environment is ready for training! Next:
1. Design policy network (Step 3)
2. Implement training algorithm (e.g., PPO, SAC)
3. Train on the environment
4. Evaluate performance

## Files

| File | Purpose |
|------|---------|
| `deformable_handover_env.py` | Main environment |
| `test_environment.py` | Full test suite |
| `verify_step2.py` | Quick verification |
| `STEP2_COMPLETION_SUMMARY.md` | Detailed documentation |

## Support

For issues:
- Check `STEP2_COMPLETION_SUMMARY.md` for detailed docs
- Run `verify_step2.py` to diagnose problems
- Review `test_environment.py` for usage examples

---

**Status**: ✅ Step 2 Complete
**Integration**: Step 1 data pipeline ✓
**Last Updated**: December 29, 2025

