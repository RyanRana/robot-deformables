# Adversarial Training - Step 4

## What Was Changed

### 1. Environment Modifications (`deformable_handover_env.py`)
- Added `adversarial_mode` parameter to environment `__init__`
- **At reset**: Applies random ±0.2m perturbation to human pose
- **During steps**: 10% chance per step to apply ±0.2m perturbation
- This creates more robust policies that can handle human movement

### 2. Training Script Updates (`train_step4_rl.py`)
- Added `adversarial_mode` parameter to `train_rl_finetuning()`
- **Increased training**: 20k → 100k total timesteps
- **Curriculum**: 5k → 20k timesteps for rigid phase
- Automatically enables adversarial mode based on first run failure

## Training Parameters

```python
total_timesteps = 100,000      # 100k as per spec
curriculum_timesteps = 20,000  # First 20k with curriculum
adversarial_mode = True        # Enabled
```

## Expected Training Time

- **With GPU (RTX 5060 Ti)**: ~3-4 hours
- **Episodes**: ~780 episodes @ 128 steps per rollout
- **Updates**: ~780 PPO updates

## What Adversarial Training Does

### Perturbations Applied:
1. **Initial pose** (at reset): Human position shifted by random ±0.2m in x/y/z
2. **Dynamic perturbations** (during episode): 10% chance per step

### Benefits:
- ✅ Handles unpredictable human movement
- ✅ More robust to positioning errors  
- ✅ Better generalization to unseen scenarios
- ✅ Reduces overfitting to fixed trajectories

## Running the Training

On your Vast.ai instance (already SSH'd):

```bash
python3 train_step4_rl.py
```

## Monitoring Progress

The training will show:
- **Step N/100000**: Progress through training
- **Avg Reward**: Should improve from negative to positive
- **Loss**: PPO loss (will fluctuate)

## Expected Results

After 100k steps with adversarial training:
- **Target**: ≥85% success rate
- **Previous**: 0% (without adversarial training)
- **Improvement expected**: Significant (typically 60-80%+)

## Why This Works

1. **Curriculum Learning**: Start with easier scenarios
2. **Adversarial Robustness**: Handle unexpected situations
3. **More Training**: 5x more timesteps (20k → 100k)
4. **Hybrid IL+RL**: Combines demonstration learning with exploration

## Cost Estimate

- **Time**: ~3-4 hours
- **Rate**: $0.072/hour
- **Total**: ~$0.22-$0.29

## Decision Criterion

Per Step 4 spec:
- ✅ **Pass**: Success rate ≥ 85%
- ⚠️ **Marginal**: 75% ≤ Success < 85%
- ❌ **Fail**: Success < 75% (retry with more steps)

## Files Modified

1. `deformable_handover_env.py` - Added adversarial perturbations
2. `train_step4_rl.py` - Enabled 100k step adversarial training

---

**Status**: Ready to train! Just run the script in your Vast.ai terminal.

