# Breaking the "Hold Still Forever" Plateau - Step 4

## Problem Identified

Debug episode showed the policy learned a **safe but suboptimal exploit**:
- ✅ Successfully grasps towel
- ✅ Holds it stably at 1m height
- ✅ Maintains gentle 1-2N force
- ✅ Collects +0.46 reward/step for 100 steps = +46 total
- ❌ **BUT**: Never moves toward human (stays 0.731m away)
- ❌ **Never attempts handover!**

**Root cause**: The reward function made "holding still" too comfortable and didn't incentivize forward progress.

## Fixes Implemented

### 1. **Strong Progress Reward** ✅ (Most Important)

**Added distance progress tracking**:
```python
# Track how much closer we got to human this step
progress = prev_distance - current_distance
reward += 2.0 * progress  # Big reward for moving closer!

# Also penalize being far away
distance_penalty = -0.3 * arm_to_human_dist
reward += distance_penalty

# Bonus for being very close
if distance < 0.3m: reward += 1.0
if distance < 0.5m: reward += 0.5
```

**Impact**: Now "holding still" loses reward over time, and moving toward human gains reward!

### 2. **Reduced Holding Bonuses** ✅

**Before** (too comfortable):
```python
+ 0.2 for grasping
+ 0.1 for height
+ 0.1 for gentle force
= +0.4 per step just for existing
```

**After** (minimal):
```python
+ 0.05 for grasping  (reduced 4x)
+ 0.02 for height    (reduced 5x)
+ 0.05 for gentle    (reduced 2x)
= +0.12 per step baseline
```

**Impact**: Can't coast on easy rewards anymore!

### 3. **Increased Contact Rewards** ✅

```python
if contact_force > 0: reward += 0.5  (was 0.2)
if contact_force > 2: reward += 1.0  (new!)
```

**Impact**: Big bonus for actually touching the human!

### 4. **Time Pressure** ✅

```python
time_penalty = -0.02 * (step_count / max_steps)
```

**Impact**: Penalty increases as episode progresses - can't stall forever!

### 5. **Shorter Episodes** ✅

- Changed: `max_steps = 70` (was 100)
- **Impact**: Forces faster action, no time to coast

### 6. **More Exploration** ✅

- Increased entropy bonus: `entropy_coef = 0.05` (was 0.01)
- Actually computed entropy from policy std
- **Impact**: Encourages trying new actions instead of being stuck

### 7. **Bigger Success Bonus** ✅

- Success: `+50` (was already good)
- This makes the risk of moving worth it!

## Expected Behavior Change

### Old Behavior (Plateau):
```
Step 0: Grasp towel → +rewards
Step 1-99: Hold perfectly still at 0.73m away
  → +0.46 per step × 100 = +46 total
Episode ends by timeout, no handover attempted
```

### New Behavior (Progress):
```
Step 0-10: Grasp towel → +small rewards
Step 11: Move 0.05m closer to human
  → +2.0 * 0.05 (progress) + distance reduction = +bonus!
Step 12: Keep moving closer
  → More progress rewards
Step 20: Within 0.5m of human
  → +0.5 proximity bonus
Step 30: Within 0.3m, contact_force > 0
  → +1.0 proximity + 0.5 contact = +1.5 bonus!
Step 35: Strong contact (handover)
  → +50 success bonus = HIGH TOTAL REWARD
```

## Reward Breakdown Comparison

### "Hold Still" Strategy (OLD reward function):
```
100 steps × 0.46 = +46.0 total
Safe, but no handover ❌
```

### "Hold Still" Strategy (NEW reward function):
```
70 steps × 0.12 (baseline)
- 0.3 × 0.73 (distance penalty) × 70
- 0.02 × increasing time penalty
≈ +8.4 - 15.3 - 0.7 = -7.6 total
Now PUNISHED for holding still! ✅
```

### "Move Toward Human" Strategy (NEW reward function):
```
10 steps setup: +1.2
30 steps moving (0.02m/step progress):
  +2.0 × 0.02 × 30 = +12.0 (progress)
  +distance reduction bonus ≈ +3.0
10 steps near human (<0.5m):
  +0.5 × 10 = +5.0
5 steps in contact:
  +0.5 × 5 = +2.5
Final handover: +50.0
Total ≈ +73.7 ✅✅✅

VS old "hold still" reward of +46!
```

## Key Insight

**The exploit was rational under the old reward!**
- Old: +46 for safe holding > risk of trying handover
- New: -7.6 for holding < +73.7 for handover

**Now the optimal strategy is to actually complete the task!**

## Files Modified

1. `deformable_handover_env.py`:
   - Added progress reward (distance tracking)
   - Reduced holding bonuses
   - Increased contact rewards
   - Added time pressure
   - Shortened episodes (70 steps)

2. `train_step4_rl.py`:
   - Increased entropy coefficient (0.05)
   - Computed actual entropy loss
   - More exploration encouraged

## How to Verify

### Run debug episode again:
```bash
python3 debug_single_episode.py
```

**Look for**:
- ✅ Distance to human DECREASING over time (not staying at 0.73m!)
- ✅ Higher final reward (> +46)
- ✅ More varied behavior (exploration)

### Run training:
```bash
python3 train_step4_rl.py
```

**Look for**:
- ✅ Episodes getting shorter (successful handovers < 70 steps)
- ✅ Higher rewards than before (+50-70 range)
- ✅ Success count increasing

## Summary

The policy was stuck because:
1. It found a local optimum (+46 for holding)
2. Moving toward human seemed risky (might drop, lose reward)
3. No incentive to try handover

Now:
1. Holding still is PUNISHED (-7.6)
2. Moving closer is REWARDED (+progress bonus)
3. Handover is VERY rewarding (+73.7)

**The gradient now points toward the correct behavior!** 🎯

---

**Status**: Ready to break the plateau! Upload and run! 🚀

