# Curriculum Learning - The Best Approach for Step 4

## Why Previous Attempt Failed (1% Success at 500k Steps)

### The Local Optimum Problem
The policy learned to **exploit the reward function**:
- Got +17 reward by: holding towel + being near human + timeout
- **Without actually completing the handover!**
- This was safer and easier than risking a failed handover attempt

### Why It Happened
1. **Too many passive rewards**: holding (+0.05), height (+0.02), gentle force (+0.05), proximity (+0.5-1.0)
2. **Success bonus too small**: +50 wasn't enough to outweigh the safe +17 strategy
3. **Task too hard from start**: 0.7m distance is challenging for initial learning

## The Solution: Curriculum Learning + Sparse Rewards

### 3-Stage Curriculum

**Stage 1: EASY (0.25m) - 100k steps**
- Human very close to robot
- Easy to complete handover
- Builds confidence and learns basic mechanics
- Target: 70-80% success

**Stage 2: MEDIUM (0.45m) - 150k steps**  
- Human at medium distance
- Transfers Stage 1 skills
- Learns to reach further
- Target: 50-70% success

**Stage 3: HARD (0.7m+) - 200k steps**
- Full task difficulty
- Applies learned skills to hard case
- Final fine-tuning
- Target: 75-85% success

### Sparse Reward Design

**Removed all passive rewards**:
- ❌ No more holding bonus
- ❌ No more height reward
- ❌ No more gentle force reward  
- ❌ No more proximity bonuses

**Kept only task-relevant rewards**:
- ✅ Progress reward: +5.0 × (distance_decreased) - ONLY way to get positive rewards!
- ✅ Contact reward: +1.0 if touching human, +2.0 if strong contact
- ✅ Success bonus: **+150** (increased from +50) - Makes handover VERY valuable
- ✅ Constant penalty: -0.1 per step - Encourages fast completion
- ✅ Failure penalties: -20 for drop/force (moderate)

### Why This Will Work

**1. No More Exploits**
Without passive rewards, the ONLY way to get positive total reward is:
- Make progress toward human: +5 per 0.1m = +35 total for 0.7m
- Complete handover: +150
- **Total: +185 for success** vs **-7 for timeout**

**2. Curriculum Builds Skills**
- Stage 1: Learn "what is a handover?" (easy version)
- Stage 2: Learn "how to reach further?"
- Stage 3: Apply to full task

**3. Skill Transfer**
The policy learns the handover strategy in Stage 1, then just adapts the reaching distance in Stages 2-3. Much more efficient than learning from scratch!

## Training Plan

### Timeline
- **Stage 1**: ~2-3 hours
- **Stage 2**: ~3-4 hours
- **Stage 3**: ~4-5 hours
- **Total**: ~10-12 hours

### Cost
- ~$0.75-$0.90 @ $0.072/hr
- Much less than another 500k random training ($1.30)

### Expected Results

| Stage | Distance | Steps | Expected Success |
|-------|----------|-------|------------------|
| 1 | 0.25m | 100k | 70-80% |
| 2 | 0.45m | 150k | 50-70% |
| 3 | 0.7m+ | 200k | **75-85%** |

## Why This is Better Than Alternatives

### vs. More Random Training (1M steps)
- ❌ Might find another exploit
- ❌ Costs 2x more time and money
- ❌ No guarantee of success

### vs. Sparse Rewards Only (no curriculum)
- ❌ Very hard to learn from scratch with sparse rewards
- ❌ Would need 500k+ steps just to discover basic handover

### vs. Just Increase Success Bonus
- ❌ Doesn't address the "task too hard" problem
- ❌ Policy still might not discover successful handovers

## Implementation

### Key Files Modified

1. **`deformable_handover_env.py`**:
   - Added `human_distance` parameter for curriculum
   - Removed all passive rewards (no more +17 exploit!)
   - Increased success bonus to +150
   - Only progress and contact rewards remain

2. **`train_step4_curriculum.py`** (NEW):
   - 3-stage training script
   - Automatic stage progression
   - Evaluation after each stage
   - Transfers policy weights between stages

### How to Run

```bash
python3 train_step4_curriculum.py
```

That's it! The script handles all 3 stages automatically.

## Monitoring Progress

### Stage 1 (Easy)
```
Should see:
- Rewards: +50 to +100 range quickly
- Success rate: 60%+ after 50k steps
- Episodes: 20-40 steps (fast handovers)
```

### Stage 2 (Medium)
```
Should see:
- Initial dip in success (harder task)
- Gradual recovery to 50-60%
- Rewards: +30 to +80 range
```

### Stage 3 (Hard)
```
Should see:
- Initial success: 30-40%
- Final success: 75-85%
- Rewards: +20 to +150 range
- Consistent handovers by end
```

## If It Still Doesn't Work

If Stage 1 fails (<50% success):
- Task might be too hard even at 0.25m
- Consider starting at 0.15m
- Or add small holding bonus (+0.02) back

If Stage 3 fails (<60% success):
- Increase Stage 3 timesteps (200k → 300k)
- Or lower final distance (0.7m → 0.6m)
- Or increase success bonus (+150 → +200)

## Confidence Level

**HIGH** - This approach has the best chance of success because:
1. ✅ Curriculum learning is proven effective for hard tasks
2. ✅ Sparse rewards prevent reward hacking
3. ✅ High success bonus (+150) makes handover clearly optimal
4. ✅ Reasonable training time and cost

The 1% failure wasn't due to bad model architecture or insufficient training time - it was due to reward design allowing exploitation. This fixes that core issue.

---

**Ready to train!** 🎓🚀

