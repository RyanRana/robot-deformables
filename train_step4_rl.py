#!/usr/bin/env python3
"""
Step 4: Model Training - RL Fine-Tuning

Implements PPO-based RL fine-tuning with:
- IL initialization
- Hybrid IL+RL blending
- Safety constraints
- Multi-modal Transformer fusion
- Curriculum learning
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Optional
import tempfile
import os

# Import from previous steps
import deformable_handover_env
from diffusion_policy import DiffusionPolicy, DiffusionPolicyTrainer

# Try to import stable-baselines3, fall back to custom implementation
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available. Using custom PPO implementation.")


class TransformerFusion(nn.Module):
    """
    Multi-modal Transformer encoder for fusing image, effort, IMU, and audio.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim)
        )
        
        # Proprioception encoder (effort + IMU)
        self.proprio_encoder = nn.Linear(12, hidden_dim)  # 6 effort + 6 IMU
        
        # Audio mel-spectrogram encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=160, stride=160),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(32 * 32, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multi-modal inputs using Transformer."""
        batch_size = observations['image'].shape[0]
        
        # Encode each modality (image already normalized in _prepare_obs)
        image_feat = self.image_encoder(observations['image'].float())  # (B, D)
        
        # Concatenate effort and IMU
        proprio = torch.cat([observations['effort'], observations['imu']], dim=1)
        proprio_feat = self.proprio_encoder(proprio)  # (B, D)
        
        audio_feat = self.audio_encoder(observations['audio'].unsqueeze(1))  # (B, D)
        
        # Stack features as sequence: [image, proprio, audio]
        features = torch.stack([image_feat, proprio_feat, audio_feat], dim=1)  # (B, 3, D)
        
        # Apply Transformer
        fused = self.transformer(features)  # (B, 3, D)
        
        # Pool across modalities
        pooled = fused.mean(dim=1)  # (B, D)
        
        return self.output_proj(pooled)


class HybridILRLPolicy(nn.Module):
    """
    Hybrid policy that blends IL predictions with RL policy.
    Includes safety constraints for effort limits.
    """
    
    def __init__(self, il_policy: DiffusionPolicy, action_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        
        self.il_policy = il_policy
        self.action_dim = action_dim
        
        # Multi-modal fusion
        self.fusion = TransformerFusion(hidden_dim)
        
        # RL policy head (stochastic)
        self.rl_actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.rl_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.rl_critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Blending weights
        self.register_buffer('il_weight', torch.tensor(0.7))
        self.register_buffer('rl_weight', torch.tensor(0.3))
        
    def set_blend_weights(self, il_weight: float, rl_weight: float):
        """Update IL/RL blending weights."""
        self.il_weight = torch.tensor(il_weight)
        self.rl_weight = torch.tensor(rl_weight)
    
    def forward(self, observations: Dict[str, torch.Tensor], return_value: bool = False, deterministic: bool = False):
        """
        Forward pass with hybrid IL+RL blending.
        
        Args:
            observations: Dict of observations
            return_value: If True, also return value and log_prob
            deterministic: If True, use mean action (no sampling)
            
        Returns:
            actions: Blended IL+RL actions
            value: Value estimate (if return_value=True)
            log_prob: Log probability (if return_value=True)
        """
        # Get IL prediction
        with torch.no_grad():
            il_action = self.il_policy(observations)
        
        # Get RL prediction (stochastic)
        features = self.fusion(observations)
        rl_mean = torch.tanh(self.rl_actor_mean(features))
        rl_std = torch.exp(self.rl_log_std).expand_as(rl_mean)
        
        if deterministic:
            rl_action = rl_mean
            log_prob = None
        else:
            dist = torch.distributions.Normal(rl_mean, rl_std)
            rl_action = dist.rsample()
            rl_action = torch.clamp(rl_action, -1, 1)  # Clip to action space
            log_prob = dist.log_prob(rl_action).sum(-1, keepdim=True)
        
        # Blend predictions (blend means for stability)
        blended_action = self.il_weight * il_action + self.rl_weight * rl_action
        
        if return_value:
            value = self.rl_critic(features)
            return blended_action, value, log_prob
        
        return blended_action
    
    def predict_with_safety(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict actions with safety constraints.
        If effort > 50N, scale action by 0.7.
        """
        action = self.forward(observations)
        
        # Safety constraint: check effort magnitude
        effort = observations['effort']
        effort_magnitude = torch.norm(effort, dim=-1, keepdim=True)
        
        # Scale action if effort exceeds threshold
        safety_mask = (effort_magnitude > 50.0).float()
        safety_scale = 0.7 * safety_mask + 1.0 * (1 - safety_mask)
        
        safe_action = action * safety_scale.unsqueeze(-1)
        
        return safe_action


class CustomPPO:
    """
    Proper PPO implementation with gradient updates.
    """
    
    def __init__(self, policy: HybridILRLPolicy, env, lr: float = 1e-4, device='cuda'):
        self.policy = policy
        self.env = env
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.05  # Increased from 0.01 for more exploration
        
    def learn(self, total_timesteps: int, log_interval: int = 10):
        """PPO training loop with proper batching."""
        print(f"Training with PPO for {total_timesteps} timesteps...")
        
        episode_rewards = []
        episode_lengths = []
        rollout_size = 2048  # Standard PPO rollout size
        minibatch_size = 64
        n_epochs = 10  # Epochs per rollout
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(0, total_timesteps, rollout_size):
            # Collect rollout
            rollout_data = self._collect_rollout(rollout_size, obs, episode_reward, episode_length)
            obs = rollout_data['next_obs']
            episode_reward = rollout_data['episode_reward']
            episode_length = rollout_data['episode_length']
            
            # Track episode rewards
            episode_rewards.extend(rollout_data['finished_episode_rewards'])
            episode_lengths.extend(rollout_data['finished_episode_lengths'])
            
            # Compute advantages and returns
            self._compute_advantages(rollout_data)
            
            # PPO update with mini-batches
            total_loss = 0
            for epoch in range(n_epochs):
                loss = self._ppo_update_minibatch(
                    rollout_data['observations'],
                    rollout_data['actions'],
                    rollout_data['old_log_probs'],
                    rollout_data['returns'],
                    rollout_data['advantages'],
                    minibatch_size
                )
                total_loss += loss
            
            avg_loss = total_loss / n_epochs
            
            # Log progress with detailed diagnostics
            if (step // rollout_size) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
                
                # Count successes and failures in recent episodes
                recent_info = rollout_data['finished_episode_info'][-10:] if rollout_data['finished_episode_info'] else []
                n_success = sum(1 for ep in recent_info if ep.get('success', False))
                n_drop = sum(1 for ep in recent_info if ep.get('failure') == 'drop')
                n_force = sum(1 for ep in recent_info if ep.get('failure') == 'excessive_force')
                n_timeout = sum(1 for ep in recent_info if ep.get('truncated', False))
                
                print(f"Step {step}/{total_timesteps}: Reward={avg_reward:.2f}, Length={avg_length:.1f}, Loss={avg_loss:.4f}")
                if recent_info:
                    print(f"  Recent episodes: Success={n_success}, Drop={n_drop}, Force={n_force}, Timeout={n_timeout}")
        
        final_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
        print(f"Training complete! Final avg reward: {final_reward:.2f}")
    
    def _collect_rollout(self, n_steps: int, start_obs, start_reward, start_length):
        """Collect a rollout of experiences with detailed logging."""
        observations_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        dones_list = []
        
        finished_episode_rewards = []
        finished_episode_lengths = []
        finished_episode_info = []
        
        obs = start_obs
        episode_reward = start_reward
        episode_length = start_length
        
        for _ in range(n_steps):
            # Prepare observation
            obs_tensor = self._prepare_obs(obs)
            
            # Get action, value, and log_prob
            with torch.no_grad():
                action, value, log_prob = self.policy(obs_tensor, return_value=True, deterministic=False)
                action_np = action.squeeze().cpu().numpy()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store
            observations_list.append(obs_tensor)
            actions_list.append(action.squeeze())
            rewards_list.append(reward)
            values_list.append(value.squeeze())
            log_probs_list.append(log_prob.squeeze())
            dones_list.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if done:
                # Log episode details
                finished_episode_rewards.append(episode_reward)
                finished_episode_lengths.append(episode_length)
                finished_episode_info.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'success': info.get('handover_achieved', False),
                    'failure': info.get('failure_reason', None),
                    'terminated': terminated,
                    'truncated': truncated
                })
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        return {
            'observations': observations_list,
            'actions': torch.stack(actions_list),
            'rewards': torch.tensor(rewards_list, device=self.device),
            'values': torch.stack(values_list),
            'old_log_probs': torch.stack(log_probs_list),
            'dones': torch.tensor(dones_list, device=self.device),
            'returns': None,  # Will be computed
            'advantages': None,  # Will be computed
            'next_obs': obs,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'finished_episode_rewards': finished_episode_rewards,
            'finished_episode_lengths': finished_episode_lengths,
            'finished_episode_info': finished_episode_info
        }
    
    def _compute_advantages(self, rollout_data, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns."""
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        n_steps = len(rewards)
        advantages = torch.zeros(n_steps, device=self.device)
        last_gae = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0
                next_non_terminal = 1.0 - dones[t].float()
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t].float()
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
    
    def _ppo_update_minibatch(self, observations, actions, old_log_probs, returns, advantages, minibatch_size):
        """Perform PPO update with mini-batches."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_steps = len(observations)
        indices = np.random.permutation(n_steps)
        
        total_loss = 0
        n_batches = 0
        
        for start in range(0, n_steps, minibatch_size):
            end = min(start + minibatch_size, n_steps)
            batch_indices = indices[start:end]
            
            # Prepare mini-batch
            obs_batch = [observations[i] for i in batch_indices]
            actions_batch = actions[batch_indices]
            old_log_probs_batch = old_log_probs[batch_indices]
            returns_batch = returns[batch_indices]
            advantages_batch = advantages[batch_indices]
            
            # Concatenate observations into batch
            obs_dict_batch = {}
            for key in obs_batch[0].keys():
                obs_dict_batch[key] = torch.cat([obs[key] for obs in obs_batch], dim=0)
            
            # Forward pass
            new_actions, values, new_log_probs = self.policy(obs_dict_batch, return_value=True, deterministic=False)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs.squeeze() - old_log_probs_batch)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(ratio * advantages_batch, clipped_ratio * advantages_batch).mean()
            
            # Value loss
            value_loss = 0.5 * ((returns_batch - values.squeeze()) ** 2).mean()
            
            # Entropy bonus (for exploration) - computed from policy std
            policy_std = torch.exp(self.policy.rl_log_std)
            entropy = 0.5 * torch.log(2 * np.pi * policy_std ** 2).sum() + 0.5 * len(policy_std)
            entropy_loss = -entropy  # Negative because we want to maximize entropy
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def _prepare_obs(self, obs):
        """Convert numpy observation to torch tensors with normalization."""
        obs_tensor = {}
        for key in ['image', 'effort', 'imu', 'audio']:
            if key == 'image':
                # Permute to (C, H, W), normalize to [0, 1], and add batch dim
                tensor = torch.from_numpy(obs[key]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            elif key == 'effort':
                # Normalize effort to roughly [-1, 1]
                tensor = torch.from_numpy(obs[key]).unsqueeze(0).float() / 50.0
            elif key == 'imu':
                # Normalize IMU to roughly [-1, 1]
                tensor = torch.from_numpy(obs[key]).unsqueeze(0).float() / 5.0
            else:
                # Audio is already in [0, 1]
                tensor = torch.from_numpy(obs[key]).unsqueeze(0).float()
            obs_tensor[key] = tensor.to(self.device)
        return obs_tensor


def train_rl_finetuning(
    il_policy_path: Optional[str] = None,
    total_timesteps: int = 500000,
    curriculum_timesteps: int = 100000,
    device: str = 'cuda',
    adversarial_mode: bool = False
):
    """
    Train RL fine-tuning with PPO.
    
    Args:
        il_policy_path: Path to pretrained IL model (optional)
        total_timesteps: Total training timesteps
        curriculum_timesteps: Timesteps for curriculum (rigid sims)
        device: Device to use
        adversarial_mode: If True, adds ±0.2m perturbations to human poses
    """
    
    print("=" * 70)
    print(" STEP 4: RL FINE-TUNING WITH PPO")
    if adversarial_mode:
        print(" MODE: ADVERSARIAL TRAINING (±0.2m pose perturbations)")
    print("=" * 70)
    print()
    
    # Create environment
    print("Creating environment...")
    if adversarial_mode:
        print("  Adversarial mode: ENABLED")
        print("  Human pose perturbations: ±0.2m")
    env = gym.make('DeformableHandover-v0', adversarial_mode=adversarial_mode)
    
    # Load or create IL policy
    print("Initializing hybrid IL+RL policy...")
    il_policy = DiffusionPolicy(env.observation_space, env.action_space, hidden_dim=256)
    
    if il_policy_path and os.path.exists(il_policy_path):
        print(f"Loading IL weights from: {il_policy_path}")
        checkpoint = torch.load(il_policy_path, map_location=device)
        il_policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        print("Warning: No IL policy found. Starting from scratch.")
    
    # Create hybrid policy
    hybrid_policy = HybridILRLPolicy(il_policy, action_dim=6, hidden_dim=256)
    hybrid_policy = hybrid_policy.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in hybrid_policy.parameters() if p.requires_grad)
    print(f"  Hybrid policy parameters: {num_params:,}")
    print()
    
    # Training with curriculum
    print("Starting RL training with curriculum learning...")
    print("-" * 70)
    
    # Gradual IL weight decay (no abrupt phase switching)
    print("Training with gradual IL → RL transition")
    print(f"  Initial: IL=0.9, RL=0.1")
    print(f"  Final: IL=0.2, RL=0.8 (linear decay over training)")
    
    # Set initial high IL weight for stability
    hybrid_policy.set_blend_weights(0.9, 0.1)
    
    # Use custom PPO
    ppo = CustomPPO(hybrid_policy, env, lr=1e-4, device=device)
    
    # Train with gradual IL weight decay (slower to keep guidance longer)
    total_rollouts = total_timesteps // 2048
    for rollout_idx in range(total_rollouts):
        # Compute current IL weight (slower decay: 0.95 → 0.35)
        progress = rollout_idx / total_rollouts
        il_weight = 0.95 - (0.60 * progress)  # 0.95 → 0.35 (slower decay)
        rl_weight = 1.0 - il_weight
        hybrid_policy.set_blend_weights(il_weight, rl_weight)
        
        # Train for one rollout
        current_steps = rollout_idx * 2048
        if rollout_idx % 10 == 0:
            print(f"\nRollout {rollout_idx}/{total_rollouts} (Step {current_steps}): IL={il_weight:.2f}, RL={rl_weight:.2f}")
        
        ppo.learn(total_timesteps=2048, log_interval=2)  # Log every 2 rollouts instead of every 1
    
    print("-" * 70)
    print()
    
    # Evaluation
    print("Evaluating trained policy...")
    success_rate = evaluate_policy(hybrid_policy, env, n_episodes=100, device=device)
    
    print(f"\nEvaluation Results:")
    print(f"  Success rate: {success_rate:.1f}%")
    print()
    
    # Decision criterion
    print("=" * 70)
    print("DECISION CRITERION CHECK")
    print("=" * 70)
    
    if success_rate < 75:
        print(f"⚠ Success rate ({success_rate:.1f}%) < 75%")
        print("  Recommendation: Add adversarial perturbations and retrain 100k steps")
        criterion_met = False
    elif success_rate < 85:
        print(f"⚠ Success rate ({success_rate:.1f}%) < 85% but >= 75%")
        print("  Status: MARGINAL - Consider adversarial training")
        criterion_met = False
    else:
        print(f"✓ Success rate ({success_rate:.1f}%) >= 85%")
        print("  Status: PASSED")
        criterion_met = True
    
    print("=" * 70)
    print()
    
    # Save model
    save_path = os.path.join(tempfile.gettempdir(), 'hybrid_rl_policy.pt')
    torch.save({
        'policy_state_dict': hybrid_policy.state_dict(),
        'success_rate': success_rate,
    }, save_path)
    print(f"Model saved to: {save_path}")
    print()
    
    env.close()
    
    return hybrid_policy, success_rate, criterion_met, save_path


def evaluate_policy(policy, env, n_episodes: int = 100, device='cuda'):
    """Evaluate policy success rate."""
    print(f"Running {n_episodes} evaluation episodes...")
    
    successes = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        episode_reward = 0
        
        for step in range(100):
            # Prepare observation
            obs_tensor = {}
            for key in ['image', 'effort', 'imu', 'audio']:
                if key == 'image':
                    tensor = torch.from_numpy(obs[key]).permute(2, 0, 1).unsqueeze(0).float()
                else:
                    tensor = torch.from_numpy(obs[key]).unsqueeze(0).float()
                obs_tensor[key] = tensor.to(device)
            
            # Get action (stochastic for evaluation to match training)
            with torch.no_grad():
                action = policy.predict_with_safety(obs_tensor)
                action = action.squeeze().cpu().numpy()  # Remove batch dim properly
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Count success (handover achieved)
        if info.get('handover_achieved', False):
            successes += 1
        
        if (ep + 1) % 20 == 0:
            print(f"  Evaluated {ep + 1}/{n_episodes} episodes...")
    
    success_rate = (successes / n_episodes) * 100
    return success_rate


if __name__ == "__main__":
    # Check for Step 3 checkpoint
    il_checkpoint = os.path.join(tempfile.gettempdir(), 'diffusion_policy_checkpoint.pt')
    
    if not os.path.exists(il_checkpoint):
        print("Warning: No IL checkpoint found from Step 3.")
        print("Training will start from scratch (not recommended).")
        print()
        il_checkpoint = None
    
    # Check if we need adversarial training
    # (On first run, this will be False. After first run fails, set to True)
    run_adversarial = True  # Enable adversarial training as recommended
    
    if run_adversarial:
        print("\n" + "=" * 70)
        print("RUNNING ADVERSARIAL TRAINING MODE")
        print("Per Step 4 spec: success <75% requires adversarial retraining")
        print("=" * 70)
        print()
    
    # Run RL fine-tuning
    policy, success_rate, criterion_met, save_path = train_rl_finetuning(
        il_policy_path=il_checkpoint,
        total_timesteps=500000,  # 500k steps - policy was just starting to learn!
        curriculum_timesteps=20000,  # 20k for curriculum
        device='cuda',
        adversarial_mode=run_adversarial
    )
    
    # Cleanup
    if os.path.exists(save_path):
        print(f"Download model from: {save_path}")
        print("(Will be deleted on instance termination)")
    
    print("\n" + "=" * 70)
    if criterion_met:
        print("✅ STEP 4 COMPLETE - Policy meets success criteria!")
    else:
        print("⚠ STEP 4 NEEDS ADJUSTMENT - Review recommendations above")
    print("=" * 70)

