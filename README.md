# Collaborative Robot - Deformable Object Handover with Multi-Modal RL

A complete implementation of a robotic handover system using imitation learning and reinforcement learning, featuring multi-modal sensor fusion (vision, proprioception, audio) and deformable object manipulation.

## Project Overview

This project implements a 4-step pipeline for training a robot to perform deformable object handovers:

1. **Data Preparation**: Streaming data pipeline from ALOHA and FrodoBots datasets
2. **Environment Setup**: Custom Gymnasium environment with PyBullet physics
3. **Imitation Learning**: DiffusionPolicy pretraining on expert demonstrations
4. **RL Fine-Tuning**: PPO-based fine-tuning with hybrid IL+RL policy

### Key Features

- ✅ **Streaming data loading** from HuggingFace datasets (125k+ frames)
- ✅ **Multi-modal observations**: RGB images, force/torque, IMU, audio waveforms
- ✅ **Custom Gymnasium environment** with deformable object physics
- ✅ **Hybrid IL+RL architecture** with Transformer fusion
- ✅ **Proper PPO implementation** with GAE, stochastic policies, mini-batch updates
- ✅ **Reward shaping** and curriculum learning strategies
- ✅ **GPU-accelerated training** on cloud infrastructure (Vast.ai)

## Architecture

### Neural Network Architecture

```
Multi-Modal Encoder (Transformer Fusion)
├── Image Encoder: CNN (3x84x84 → 256)
├── Proprioception Encoder: MLP (effort + IMU → 256)
└── Audio Encoder: 1D CNN + Adaptive Pooling (16kHz → 256)
    ↓
Transformer Encoder (3 layers, 8 heads)
    ↓
Policy Head (Stochastic)
├── Mean Network: MLP (256 → 6)
└── Learned Log-Std: Parameter (6)
    ↓
6DoF End-Effector Control
```

### Environment

- **Observation Space**: Dict with image (84×84×3), effort (6), IMU (6), audio (16000)
- **Action Space**: Box(6) - normalized 6DoF end-effector velocities
- **Physics**: PyBullet soft-body simulation for deformable towel
- **Reward**: Progress-based with contact detection and success bonus

## Project Structure

```
collaborative-robot/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data_preparation.py                # Step 1: Data pipeline
├── deformable_handover_env.py         # Step 2: Gymnasium environment  
├── diffusion_policy.py                # Step 3: IL policy architecture
├── train_step3.py                     # Step 3: IL training script
├── train_step4_rl.py                  # Step 4: RL fine-tuning
├── train_step4_curriculum.py          # Step 4: Curriculum learning
│
├── verify_step1.py                    # Verification scripts
├── verify_step2.py
├── test_environment.py
├── integration_test.py
├── debug_single_episode.py
│
└── docs/                              # Documentation
    ├── STEP1_COMPLETION_SUMMARY.md
    ├── STEP2_COMPLETION_SUMMARY.md
    ├── STEP2_QUICKSTART.md
    ├── REWARD_SHAPING_FIXES.md
    ├── PPO_FIXES_SUMMARY.md
    ├── PLATEAU_BREAKING_FIXES.md
    ├── FINAL_TRAINING_FIXES.md
    └── CURRICULUM_LEARNING_APPROACH.md
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/collaborative-robot.git
cd collaborative-robot

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preparation

```bash
python3 data_preparation.py
```

Loads and augments ALOHA dataset, creates streaming generator yielding batches of observations and actions.

**Output**: Validated data pipeline with 125k+ frames ready for training.

### Step 2: Environment Setup

```bash
python3 verify_step2.py
```

Creates custom Gymnasium environment with:
- Multi-modal observations
- Deformable object physics
- Reward shaping
- Safety constraints

**Output**: Working environment with 3-5 average reward.

### Step 3: Imitation Learning Pretraining

```bash
python3 train_step3.py
```

Trains DiffusionPolicy on expert demonstrations:
- 100 epochs, batch size 64
- Adam optimizer, lr=1e-4
- MSE loss on action predictions

**Output**: IL checkpoint with validation MSE ≤ 0.15 ✅

**Achieved**: MSE = 0.027 (validation)

### Step 4: RL Fine-Tuning

```bash
# Standard training
python3 train_step4_rl.py

# Curriculum learning
python3 train_step4_curriculum.py
```

PPO-based fine-tuning with:
- Hybrid IL+RL policy blending
- Multi-modal Transformer fusion
- Safety constraints
- Curriculum learning (3 stages)

**Target**: 85% success rate on handover task

**Achieved**: See [Results](#results) section

## Results

### What Works ✅

1. **Data Pipeline**: Successfully loads and augments 125k frames from HuggingFace datasets
2. **IL Training**: Achieved 0.027 validation MSE (target: ≤0.15)
3. **Environment**: Stable simulation with multi-modal observations
4. **Policy Architecture**: Proper Transformer fusion of vision, proprioception, and audio
5. **PPO Implementation**: Stable training with proper GAE, stochastic policies, mini-batching

### Challenges Encountered ⚠️

**Reward Engineering**: The primary challenge was reward function design:

- **Dense Rewards**: Policy found exploit - achieved +17 reward without completing handover by "holding still near human"
- **Sparse Rewards**: Policy failed to learn - 0% success rate as no gradient signal
- **Curriculum Learning**: Insufficient for overcoming sparse reward challenge

**Training Results**:
- 500k steps (dense rewards): 3% success (plateau at local optimum)
- 500k steps (sparse rewards): 0% success (no learning)
- 450k steps (curriculum): 0% success across all stages

### Key Insights

1. **Task Difficulty**: Deformable object manipulation with multi-modal RL is at the edge of tractability with current methods
2. **Reward Hacking**: Dense reward shaping is susceptible to exploitation
3. **Exploration**: Sparse rewards require prohibitive exploration in high-dimensional spaces
4. **IL Foundation**: Strong IL initialization (0.027 MSE) wasn't sufficient to bootstrap RL learning

## Technical Achievements

Despite not achieving the target success rate, this project demonstrates:

### 1. Production-Grade Data Pipeline
- Streaming data loading for large datasets
- On-the-fly augmentation (5x multiplier)
- Memory-efficient generator pattern
- Proper train/val split

### 2. Proper RL Implementation
- Stochastic policies with learned variance
- True Gaussian log probabilities (not heuristics)
- GAE with proper episode boundary handling
- Mini-batch PPO updates
- Gradient clipping and normalization

### 3. Multi-Modal Fusion
- Transformer-based sensor fusion
- Proper normalization for different modalities
- Audio processing with adaptive pooling
- Efficient batched operations

### 4. Iterative Debugging Process
- Fixed audio encoder dimension mismatches
- Resolved action space shape issues
- Corrected observation normalization
- Addressed reward exploitation

## Lessons Learned

### What Would Help

1. **Simpler Task First**: Start with rigid objects before deformable
2. **Shorter Horizon**: 30 steps instead of 70
3. **Denser Success Signal**: Multiple levels of partial credit
4. **Demonstration Quality**: More diverse handover demonstrations
5. **Model-Based Components**: Hybrid model-free + model-based approach

### Best Practices Demonstrated

- ✅ Comprehensive logging and diagnostics
- ✅ Incremental verification at each step
- ✅ Proper tensor shape handling
- ✅ Cloud GPU utilization
- ✅ Documentation throughout development

## Future Directions

### Immediate Next Steps

1. **Simplify Task**: Rigid object handover (remove deformability)
2. **Increase Demonstrations**: Collect more diverse IL data
3. **Hybrid Approaches**: Add model predictive control
4. **Reduce Horizon**: 30-step episodes
5. **Better Success Detection**: Multiple handover stages

### Research Directions

1. **Inverse RL**: Learn reward function from demonstrations
2. **Hierarchical RL**: Separate grasping and reaching skills
3. **Sim-to-Real**: Domain randomization and reality gap bridging
4. **Multi-Task Learning**: Pretrain on related tasks

## Citation

If you use this code in your research, please cite:

```bibtex
@software{collaborative_robot_2025,
  title={Collaborative Robot: Deformable Object Handover with Multi-Modal RL},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/collaborative-robot}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- ALOHA dataset for demonstration data
- LeRobot framework for policy architecture inspiration
- Stable-Baselines3 for RL reference implementation
- PyBullet for physics simulation

## Contact

For questions or collaboration opportunities:
- GitHub Issues: [github.com/yourusername/collaborative-robot/issues](https://github.com/yourusername/collaborative-robot/issues)
- Email: your.email@example.com

---

**Project Status**: Complete implementation with documented challenges and insights. Suitable for educational purposes, research baselines, and further development.

**Last Updated**: December 2025
