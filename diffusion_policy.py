"""
Diffusion Policy implementation for robotic manipulation.
Simplified version adapted for the deformable handover task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class DiffusionPolicy(nn.Module):
    """
    Simplified Diffusion Policy for action prediction.
    
    Uses a conditional denoising network to predict actions from observations.
    For simplicity, this implements a direct regression baseline (not full diffusion).
    """
    
    def __init__(self, observation_space, action_space, hidden_dim: int = 256):
        """
        Initialize the policy network.
        
        Args:
            observation_space: Gymnasium observation space (Dict)
            action_space: Gymnasium action space (Box)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.action_dim = action_space.shape[0]
        
        # Image encoder (CNN for 84x84x3 images)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU()
        )
        
        # Effort encoder
        self.effort_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # IMU encoder
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Audio encoder (simplified with adaptive pooling for robustness)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=160, stride=160),  # 16000 -> 100
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),  # Adaptive pooling to fixed size 32
            nn.Flatten(),
            nn.Linear(32 * 32, 128),  # 1024 -> 128
            nn.ReLU()
        )
        
        # Fusion network
        fusion_input_dim = hidden_dim + 64 + 64 + 128  # image + effort + imu + audio
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.action_dim)
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to predict actions from observations.
        
        Args:
            observations: Dict with keys 'image', 'effort', 'imu', 'audio'
        
        Returns:
            Predicted actions (batch_size, action_dim)
        """
        # Encode image (batch_size, 3, 84, 84)
        image = observations['image'].float() / 255.0  # Normalize to [0, 1]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image_features = self.image_encoder(image)
        
        # Encode effort (batch_size, 6)
        effort = observations['effort']
        if effort.dim() == 1:
            effort = effort.unsqueeze(0)
        effort_features = self.effort_encoder(effort)
        
        # Encode IMU (batch_size, 6)
        imu = observations['imu']
        if imu.dim() == 1:
            imu = imu.unsqueeze(0)
        imu_features = self.imu_encoder(imu)
        
        # Encode audio (batch_size, 1, 16000)
        audio = observations['audio']
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(1)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)
        audio_features = self.audio_encoder(audio)
        
        # Fuse all features
        fused = torch.cat([image_features, effort_features, imu_features, audio_features], dim=1)
        
        # Predict actions
        actions = self.fusion(fused)
        
        return torch.tanh(actions)  # Normalize to [-1, 1]


class DiffusionPolicyTrainer:
    """Trainer for DiffusionPolicy with GPU support."""
    
    def __init__(self, policy: DiffusionPolicy, device: str = 'auto', lr: float = 1e-4):
        """
        Initialize trainer.
        
        Args:
            policy: DiffusionPolicy model
            device: 'auto', 'cuda', 'mps', or 'cpu'
            lr: Learning rate
        """
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Move policy to device
        self.policy = policy.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {
            'observation': {},
            'action': torch.FloatTensor(batch['action']).to(self.device),
            'reward': torch.FloatTensor(batch['reward']).to(self.device)
        }
        
        # Move observations
        for key in ['image', 'effort', 'imu', 'audio']:
            if key == 'image':
                # Permute image from (B, H, W, C) to (B, C, H, W)
                img = torch.from_numpy(batch['observation'][key]).permute(0, 3, 1, 2)
                device_batch['observation'][key] = img.to(self.device)
            else:
                device_batch['observation'][key] = torch.FloatTensor(
                    batch['observation'][key]
                ).to(self.device)
        
        return device_batch
    
    def train_step(self, batch: Dict) -> float:
        """Single training step."""
        self.policy.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        device_batch = self.batch_to_device(batch)
        
        # Forward pass
        predicted_actions = self.policy(device_batch['observation'])
        
        # Compute loss
        loss = self.criterion(predicted_actions, device_batch['action'])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, batch: Dict) -> Tuple[float, float]:
        """Validation step."""
        self.policy.eval()
        
        with torch.no_grad():
            # Move batch to device
            device_batch = self.batch_to_device(batch)
            
            # Forward pass
            predicted_actions = self.policy(device_batch['observation'])
            
            # Compute MSE
            mse = self.criterion(predicted_actions, device_batch['action']).item()
            
            # Compute mean absolute error
            mae = F.l1_loss(predicted_actions, device_batch['action']).item()
        
        return mse, mae
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

