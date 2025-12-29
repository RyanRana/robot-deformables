# Reward Shaping & Training Fixes - Step 4

## Critical Issues Fixed

### 1. **Reward Function Completely Reshaped** ✅

**Before (Extremely Harsh)**:
```python
reward = -1.0 per step
+ 10.0 for success (rare)
- 50.0 for drop
- 50.0 for excessive force
→ Typical episode: -1 × 100 = -100 total reward
```

**After (Dense Shaping)**:
```python
# Dense rewards every step:
+ 0.2 if holding object (grasped & towel_z > 0.2m)
+ 0.1 if towel height > 0.3m
+ 0.3 if within 0.5m of human (+ 0.1 if within 1m)
+ 0.1 if gentle force (< 20N)
- 0.2 if too much force (> 40N)
+ 0.2 if human touching towel
- 0.05 step penalty (minimal)

# Terminal rewards:
+ 50.0 for successful handover (was +10)
- 10.0 for drop (was -50)
- 10.0 for excessive force (was -50, threshold raised to 60N)
```

**Impact**: Episodes now get positive rewards for good behavior, not just huge penalties for failure!

### 2. **Gradual IL→RL Transition** ✅

**Before**: Abrupt switch at 20k steps (IL: 0.7→0.3, RL: 0.3→0.7)
**After**: Linear decay over all training
- Start: IL=0.9, RL=0.1 (stable, learns from demonstrations)
- End: IL=0.2, RL=0.8 (explores, but keeps IL guidance)
- Smooth transition prevents policy collapse

### 3. **Detailed Episode Diagnostics** ✅

Now logs for every finished episode:
- Success/failure counts
- Failure reasons (drop, excessive_force, timeout)
- Episode lengths
- Rewards

Example output:
```
Step 2048/100000: Reward=5.23, Length=45.2, Loss=12.45
  Recent episodes: Success=2, Drop=3, Force=1, Timeout=4
```

### 4. **Early Termination Properly Enabled** ✅

Episodes terminate immediately on:
- Drop (towel_z < 0.1m)
- Excessive force (> 60N, increased from 50N)
- Successful handover

No more wasting 100 steps on failed episodes!

### 5. **Observation Normalization** ✅

All observations now properly normalized:
- Images: /255 → [0, 1]
- Effort: /50 → roughly [-1, 1]
- IMU: /5 → roughly [-1, 1]
- Audio: already [0, 1]

### 6. **Lower Learning Rate** ✅

Changed from 3e-4 → 1e-4 for stability

### 7. **Debug Script Added** ✅

`debug_single_episode.py` - Run single episode with step-by-step diagnostics

## Expected Results

### Before Fixes:
- Reward: -100 (always failed)
- Loss: 200-600+ (exploding)
- Episodes: All timeout or immediate failure

### After Fixes:
- Reward: Should start around -5 to +10
- Loss: 10-100 (stable)
- Episodes: Mix of outcomes, gradual improvement
- Success rate: Should increase from 0% → 60-80%+ over 100k steps

## New Reward Breakdown Example

Good episode (holding and moving toward human):
```
Step reward ≈ 0.2 (holding) + 0.1 (height) + 0.3 (near human) + 0.1 (gentle) - 0.05 (step penalty)
            = +0.65 per step
× 50 steps = +32.5
+ 50 (success) = +82.5 total ✅
```

Bad episode (drops immediately):
```
5 steps × 0.1 (trying) - 0.05 (penalty) = +0.25
+ (-10) drop penalty = -9.75 total ⚠️
```

Mediocre episode (holds but timeout):
```
100 steps × 0.3 (average) - 0.05 (penalty) = +25 total 
(much better than -100!) 📈
```

## Files Modified

1. `deformable_handover_env.py` - Complete reward reshaping
2. `train_step4_rl.py` - Gradual IL decay, detailed logging
3. `debug_single_episode.py` - New debug tool

## How to Test

### 1. Debug Single Episode
```bash
python3 debug_single_episode.py
```

### 2. Run Full Training
```bash
python3 train_step4_rl.py
```

### 3. Monitor Output
Look for:
- ✅ Positive rewards appearing
- ✅ Mix of failure types (not all drops)
- ✅ Some successes after ~10k steps
- ✅ Stable loss (<100)

## Key Insight

The previous reward was like:
> "Do nothing for 99 steps, then get -100 for failing"

The new reward is:
> "Get +0.3-0.7 for every good action, -0.05 for time, +50 for success, -10 for major failure"

This gives the policy **constant feedback** on what's good/bad, not just binary success/failure!

---

**Status**: Ready to train! These fixes address all the root causes identified. 🚀

