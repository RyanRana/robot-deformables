# Quick Start Guide

Get up and running with the Collaborative Robot project in minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but not required)
- 8GB+ RAM
- ~2GB disk space

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/collaborative-robot.git
cd collaborative-robot

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test Run

### 1. Verify Data Pipeline (Step 1)

```bash
python3 verify_step1.py
```

**Expected output**: Data loads successfully, shows batch statistics

### 2. Test Environment (Step 2)

```bash
python3 verify_step2.py
```

**Expected output**: Environment runs, average reward 3-5

### 3. Visualize Single Episode

```bash
python3 debug_single_episode.py
```

**Expected output**: Step-by-step episode breakdown with rewards and diagnostics

## Training Pipeline

### Step 3: Train IL Policy (~2 hours)

```bash
python3 train_step3.py
```

**What it does**:
- Loads ALOHA dataset
- Trains DiffusionPolicy for 100 epochs
- Saves checkpoint to `/tmp/diffusion_policy_checkpoint.pt`

**Expected result**: Validation MSE ≤ 0.15

### Step 4: RL Fine-Tuning (~18 hours)

```bash
python3 train_step4_rl.py
```

**What it does**:
- Loads IL checkpoint from Step 3
- Trains hybrid IL+RL policy with PPO
- Runs for 500k timesteps

**Note**: This is computationally intensive. Consider using cloud GPU (see below).

## Cloud GPU Training (Vast.ai)

### Setup

1. Create account at [vast.ai](https://vast.ai)
2. Add SSH key in account settings
3. Select instance:
   - GPU: RTX 3060/3090/4090 or similar
   - RAM: 16GB+
   - Storage: 20GB+
   - Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`

### Upload and Run

```bash
# On your local machine
cd collaborative-robot
scp -r -P <PORT> . root@<IP>:/workspace/

# SSH into instance
ssh -p <PORT> root@<IP>

# On remote machine
cd /workspace
pip install -r requirements.txt

# Run training
python3 train_step3.py  # ~2 hours
python3 train_step4_rl.py  # ~18 hours
```

## Understanding Results

### Step 3 (IL Training)

Success criteria: **Validation MSE ≤ 0.15**

Output example:
```
Validation MSE: 0.027 ✅
Validation MAE: 0.126
Status: PASSED
```

### Step 4 (RL Training)

Target: **85% success rate**

Output example:
```
Evaluation Results:
  Success rate: X.X%
  
Decision: 
  ✅ PASSED (≥85%) or 
  ⚠️ NEEDS ADJUSTMENT (<85%)
```

**Note**: Current implementation achieves 0-3% due to reward engineering challenges. See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for details.

## Project Structure

```
collaborative-robot/
├── README.md              # Main documentation
├── QUICK_START.md        # This file
├── PROJECT_SUMMARY.md    # Detailed project analysis
├── requirements.txt      # Python dependencies
├── LICENSE              # MIT license
│
├── Core Implementation
├── data_preparation.py       # Step 1: Data pipeline
├── deformable_handover_env.py # Step 2: Environment
├── diffusion_policy.py       # Step 3: Policy architecture
├── train_step3.py           # Step 3: IL training
├── train_step4_rl.py        # Step 4: RL training
├── train_step4_curriculum.py # Step 4: Curriculum variant
│
├── Testing & Verification
├── verify_step1.py          # Test data pipeline
├── verify_step2.py          # Test environment
├── test_environment.py      # Integration tests
├── integration_test.py      # Full pipeline test
├── debug_single_episode.py  # Episode visualization
│
└── docs/                    # Detailed documentation
    ├── STEP1_COMPLETION_SUMMARY.md
    ├── STEP2_COMPLETION_SUMMARY.md
    ├── REWARD_SHAPING_FIXES.md
    ├── PPO_FIXES_SUMMARY.md
    └── ... (9 more documents)
```

## Common Issues

### Import Errors

```bash
# If you get "ModuleNotFoundError"
pip install -r requirements.txt --upgrade
```

### GPU Out of Memory

```python
# In train_step3.py or train_step4_rl.py, reduce batch size:
batch_size = 32  # Instead of 64
```

### Data Loading Fails

```bash
# Hugging Face datasets may require login for some datasets
huggingface-cli login
```

### PyBullet Not Available

The environment falls back to a simplified physics simulator automatically. No action needed, but results may differ.

## Next Steps

1. **Read**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for detailed insights
2. **Explore**: `docs/` folder for technical deep dives
3. **Modify**: Adjust rewards in `deformable_handover_env.py`
4. **Experiment**: Try simplified tasks (rigid objects, shorter horizons)

## Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: GitHub Issues
- **Code**: Inline comments throughout

## Contributing

Contributions welcome! Areas for improvement:

1. Simplified task variants
2. Alternative reward functions
3. Additional baselines
4. Sim-to-real transfer
5. Documentation improvements

See README.md for contribution guidelines.

## Citation

If you use this code:

```bibtex
@software{collaborative_robot_2025,
  title={Collaborative Robot: Deformable Object Handover with Multi-Modal RL},
  year={2025},
  url={https://github.com/yourusername/collaborative-robot}
}
```

---

**Happy coding!** 🤖🚀

For detailed technical information, see [README.md](README.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

