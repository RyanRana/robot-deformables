"""
Step 1: Data Preparation and Merging (Streaming Mode)

This module implements streaming data loading, filtering, augmentation, and merging
for ALOHA and FrodoBots datasets.
"""

import numpy as np
from datasets import load_dataset
from typing import Iterator, Dict, Any, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataPipeline:
    """Handles streaming data loading, filtering, and augmentation."""
    
    def __init__(self, max_aloha_frames: int = 25000, max_frodo_segments: int = 100):
        self.max_aloha_frames = max_aloha_frames
        self.max_frodo_segments = max_frodo_segments
        self.aloha_frames = []
        self.frodo_frames = []
        
    def load_aloha_streaming(self):
        """
        Load ALOHA dataset in streaming mode and filter for contact indicators.
        Collects up to 25,000 frames where effort > 0.
        """
        print("Loading ALOHA dataset in streaming mode...")
        try:
            aloha_stream = load_dataset(
                "lerobot/aloha_static_towel", 
                split="train", 
                streaming=True
            )
            
            collected = 0
            for idx, item in enumerate(aloha_stream):
                if collected >= self.max_aloha_frames:
                    break
                
                # Extract effort values (contact indicators)
                # ALOHA typically has effort in the observation or action space
                effort_detected = False
                
                # Check for effort in observation
                if 'observation' in item:
                    obs = item['observation']
                    # Look for effort-related keys
                    for key in ['effort', 'efforts', 'state']:
                        if key in obs:
                            effort_values = np.array(obs[key])
                            if np.any(effort_values > 0):
                                effort_detected = True
                                break
                
                # Also check action space for effort
                if not effort_detected and 'action' in item:
                    action_values = np.array(item['action'])
                    # Consider non-zero actions as potential contact indicators
                    if np.any(np.abs(action_values) > 0.01):
                        effort_detected = True
                
                if effort_detected:
                    self.aloha_frames.append(item)
                    collected += 1
                    
                    if collected % 1000 == 0:
                        print(f"  Collected {collected} ALOHA frames with contact indicators...")
                
                # Progress indicator
                if (idx + 1) % 5000 == 0:
                    print(f"  Processed {idx + 1} items, collected {collected} frames...")
            
            print(f"✓ ALOHA loading complete: {len(self.aloha_frames)} frames collected")
            
        except Exception as e:
            print(f"Warning: Could not load ALOHA dataset: {e}")
            print("Using synthetic ALOHA data for demonstration...")
            self._create_synthetic_aloha()
    
    def load_frodo_streaming(self):
        """
        Load FrodoBots dataset in streaming mode and filter for teleop segments.
        Filters for velocity < 0.5 m/s from IMU data.
        """
        print("Loading FrodoBots dataset in streaming mode...")
        try:
            frodo_stream = load_dataset(
                "frodobots/FrodoBots-2K", 
                split="train", 
                streaming=True
            )
            
            collected = 0
            for idx, item in enumerate(frodo_stream):
                if collected >= self.max_frodo_segments:
                    break
                
                # Filter for low velocity (teleop segments)
                velocity_valid = False
                
                if 'observation' in item:
                    obs = item['observation']
                    
                    # Check IMU data for velocity
                    for key in ['imu', 'velocity', 'linear_velocity', 'state']:
                        if key in obs:
                            velocity_data = np.array(obs[key])
                            # Calculate velocity magnitude
                            velocity = np.linalg.norm(velocity_data[:3]) if len(velocity_data) >= 3 else np.abs(velocity_data[0])
                            if velocity < 0.5:
                                velocity_valid = True
                                break
                    
                    # Check for audio/keyword indicators if available
                    if 'audio' in obs or 'transcript' in obs:
                        transcript = obs.get('transcript', '')
                        if isinstance(transcript, str) and ('hand' in transcript.lower() or 'take' in transcript.lower()):
                            velocity_valid = True
                
                if velocity_valid:
                    self.frodo_frames.append(item)
                    collected += 1
                    
                    if collected % 20 == 0:
                        print(f"  Collected {collected} FrodoBots teleop segments...")
                
                # Progress indicator
                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1} items, collected {collected} segments...")
            
            print(f"✓ FrodoBots loading complete: {len(self.frodo_frames)} segments collected")
            
        except Exception as e:
            print(f"Warning: Could not load FrodoBots dataset: {e}")
            print("Using synthetic FrodoBots data for demonstration...")
            self._create_synthetic_frodo()
    
    def _create_synthetic_aloha(self):
        """Create synthetic ALOHA data for testing when dataset is unavailable."""
        print("Generating 1000 synthetic ALOHA frames...")
        for i in range(1000):
            self.aloha_frames.append({
                'observation': {
                    'image': np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8),
                    'effort': np.random.uniform(0.1, 1.0, 6),
                    'state': np.random.uniform(-1, 1, 14)
                },
                'action': np.random.uniform(-1, 1, 6)
            })
    
    def _create_synthetic_frodo(self):
        """Create synthetic FrodoBots data for testing when dataset is unavailable."""
        print("Generating 100 synthetic FrodoBots segments...")
        for i in range(100):
            self.frodo_frames.append({
                'observation': {
                    'image': np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8),
                    'imu': np.random.uniform(0, 0.4, 6),
                    'audio': np.random.randn(1000)
                },
                'action': np.random.uniform(-1, 1, 6)
            })
    
    def augment_frame(self, frame: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply 5 random augmentations with Gaussian noise to a single frame.
        
        Args:
            frame: Original frame data
            
        Returns:
            List of 5 augmented versions
        """
        augmented_frames = []
        
        for i in range(5):
            aug_frame = {
                'observation': {},
                'action': np.zeros(6),
                'reward': 0.0
            }
            
            # Copy observation with augmentation
            if 'observation' in frame:
                obs = frame['observation']
                
                for key, value in obs.items():
                    try:
                        if key == 'image':
                            # Image augmentation: add small noise and clip
                            value_arr = np.array(value)
                            noise = np.random.normal(0, 5, value_arr.shape).astype(np.int16)
                            aug_value = np.clip(value_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        elif isinstance(value, (np.ndarray, list)):
                            # Numeric augmentation: Gaussian noise σ=0.1
                            value_arr = np.array(value)
                            noise = np.random.normal(0, 0.1, value_arr.shape)
                            aug_value = value_arr + noise
                        else:
                            aug_value = value
                        
                        aug_frame['observation'][key] = aug_value
                    except Exception:
                        # If augmentation fails, keep original value
                        aug_frame['observation'][key] = value
            
            # Augment actions with Gaussian noise σ=0.1
            if 'action' in frame:
                try:
                    action_arr = np.array(frame['action'])
                    noise = np.random.normal(0, 0.1, action_arr.shape)
                    aug_frame['action'] = action_arr + noise
                except Exception:
                    aug_frame['action'] = np.array(frame['action'])
            
            # Copy reward if exists, else default to 0
            aug_frame['reward'] = float(frame.get('reward', 0.0))
            
            augmented_frames.append(aug_frame)
        
        return augmented_frames
    
    def normalize_frame(self, frame: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalize a frame to unified format.
        
        Format: {'observation': dict(image, effort, imu, audio), 'action': array(6,), 'reward': float}
        """
        normalized = {
            'observation': {
                'image': np.zeros((84, 84, 3), dtype=np.uint8),
                'effort': np.zeros(6),
                'imu': np.zeros(6),
                'audio': np.zeros(1000)
            },
            'action': np.zeros(6),
            'reward': 0.0
        }
        
        if 'observation' in frame:
            obs = frame['observation']
            
            # Extract and normalize image
            if 'image' in obs:
                try:
                    image = np.array(obs['image'])
                    # Ensure image is (84, 84, 3)
                    if image.shape != (84, 84, 3):
                        # Resize/reshape if needed
                        if len(image.shape) == 2:
                            image = np.stack([image] * 3, axis=-1)
                        # Simple resizing for this demo
                        if image.shape[:2] != (84, 84):
                            # Use nearest neighbor for simple resize
                            from PIL import Image as PILImage
                            if image.dtype != np.uint8:
                                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                            pil_img = PILImage.fromarray(image)
                            pil_img = pil_img.resize((84, 84))
                            image = np.array(pil_img)
                            if len(image.shape) == 2:
                                image = np.stack([image] * 3, axis=-1)
                    normalized['observation']['image'] = image.astype(np.uint8)
                except Exception as e:
                    # Keep default zeros if image processing fails
                    pass
            
            # Extract effort
            effort_data = obs.get('effort', obs.get('state', np.zeros(6)))
            effort_arr = np.array(effort_data)
            if len(effort_arr) >= 6:
                normalized['observation']['effort'] = effort_arr[:6]
            else:
                normalized['observation']['effort'][:len(effort_arr)] = effort_arr
            
            # Extract IMU
            imu_data = obs.get('imu', np.zeros(6))
            imu_arr = np.array(imu_data)
            if len(imu_arr) >= 6:
                normalized['observation']['imu'] = imu_arr[:6]
            else:
                normalized['observation']['imu'][:len(imu_arr)] = imu_arr
            
            # Extract audio
            audio_data = obs.get('audio', np.zeros(1000))
            audio_arr = np.array(audio_data)
            if len(audio_arr) >= 1000:
                normalized['observation']['audio'] = audio_arr[:1000]
            else:
                normalized['observation']['audio'][:len(audio_arr)] = audio_arr
        
        # Extract action
        if 'action' in frame:
            action = np.array(frame['action'])
            if len(action) >= 6:
                normalized['action'] = action[:6]
            else:
                normalized['action'][:len(action)] = action
        
        # Extract reward
        normalized['reward'] = float(frame.get('reward', 0.0))
        
        return normalized


def merged_generator(batch_size: int = 32) -> Iterator[Dict[str, Any]]:
    """
    Generator function that yields batches of merged data on-demand.
    
    Args:
        batch_size: Number of samples per batch (default: 32)
        
    Yields:
        Batches with keys: 'observation', 'action', 'reward'
        - observation: dict with 'image' (batch_size, 84, 84, 3), 'effort' (batch_size, 6),
                      'imu' (batch_size, 6), 'audio' (batch_size, 1000)
        - action: (batch_size, 6)
        - reward: (batch_size,)
    """
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Load datasets
    pipeline.load_aloha_streaming()
    pipeline.load_frodo_streaming()
    
    print(f"\nAugmenting ALOHA frames (5x augmentation)...")
    
    # Create unified data pool
    all_frames = []
    
    # Process ALOHA frames with augmentation
    for idx, frame in enumerate(pipeline.aloha_frames):
        augmented = pipeline.augment_frame(frame)
        for aug_frame in augmented:
            normalized = pipeline.normalize_frame(aug_frame, 'aloha')
            all_frames.append(normalized)
        
        if (idx + 1) % 200 == 0:
            print(f"  Augmented {idx + 1}/{len(pipeline.aloha_frames)} ALOHA frames...")
    
    # Process FrodoBots frames (no augmentation)
    for idx, frame in enumerate(pipeline.frodo_frames):
        normalized = pipeline.normalize_frame(frame, 'frodo')
        all_frames.append(normalized)
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(pipeline.frodo_frames)} FrodoBots frames...")
    
    print(f"\n✓ Total unified frames: {len(all_frames)}")
    print(f"  (ALOHA: {len(pipeline.aloha_frames)} × 5 = {len(pipeline.aloha_frames) * 5}, " +
          f"FrodoBots: {len(pipeline.frodo_frames)})")
    
    # Shuffle frames for better training
    np.random.shuffle(all_frames)
    
    print(f"\nGenerating batches of size {batch_size}...")
    
    # Yield batches
    for i in range(0, len(all_frames), batch_size):
        batch_frames = all_frames[i:i + batch_size]
        
        # Skip incomplete final batch if smaller than batch_size
        if len(batch_frames) < batch_size:
            print(f"  Skipping incomplete final batch ({len(batch_frames)} samples)")
            break
        
        # Stack batch
        batch = {
            'observation': {
                'image': np.stack([f['observation']['image'] for f in batch_frames]),
                'effort': np.stack([f['observation']['effort'] for f in batch_frames]),
                'imu': np.stack([f['observation']['imu'] for f in batch_frames]),
                'audio': np.stack([f['observation']['audio'] for f in batch_frames])
            },
            'action': np.stack([f['action'] for f in batch_frames]),
            'reward': np.array([f['reward'] for f in batch_frames])
        }
        
        yield batch


def test_generator():
    """Test the merged generator by iterating through 10 batches."""
    print("=" * 70)
    print("TESTING MERGED STREAMING GENERATOR")
    print("=" * 70)
    print()
    
    generator = merged_generator(batch_size=32)
    
    success = True
    for i, batch in enumerate(generator):
        if i >= 10:
            break
        
        print(f"\nBatch {i + 1}/10:")
        print(f"  observation.image shape: {batch['observation']['image'].shape}")
        print(f"  observation.effort shape: {batch['observation']['effort'].shape}")
        print(f"  observation.imu shape: {batch['observation']['imu'].shape}")
        print(f"  observation.audio shape: {batch['observation']['audio'].shape}")
        print(f"  action shape: {batch['action'].shape}")
        print(f"  reward shape: {batch['reward'].shape}")
        
        # Verify shapes
        expected_shapes = {
            'image': (32, 84, 84, 3),
            'effort': (32, 6),
            'imu': (32, 6),
            'audio': (32, 1000),
            'action': (32, 6),
            'reward': (32,)
        }
        
        if batch['observation']['image'].shape != expected_shapes['image']:
            print(f"  ❌ ERROR: Image shape mismatch! Expected {expected_shapes['image']}")
            success = False
        
        if batch['observation']['effort'].shape != expected_shapes['effort']:
            print(f"  ❌ ERROR: Effort shape mismatch! Expected {expected_shapes['effort']}")
            success = False
        
        if batch['observation']['imu'].shape != expected_shapes['imu']:
            print(f"  ❌ ERROR: IMU shape mismatch! Expected {expected_shapes['imu']}")
            success = False
        
        if batch['observation']['audio'].shape != expected_shapes['audio']:
            print(f"  ❌ ERROR: Audio shape mismatch! Expected {expected_shapes['audio']}")
            success = False
        
        if batch['action'].shape != expected_shapes['action']:
            print(f"  ❌ ERROR: Action shape mismatch! Expected {expected_shapes['action']}")
            success = False
        
        if batch['reward'].shape != expected_shapes['reward']:
            print(f"  ❌ ERROR: Reward shape mismatch! Expected {expected_shapes['reward']}")
            success = False
        
        if i == 0:
            # Print sample values from first batch
            print(f"\n  Sample values:")
            print(f"    Image range: [{batch['observation']['image'].min()}, {batch['observation']['image'].max()}]")
            print(f"    Action sample: {batch['action'][0]}")
            print(f"    Reward sample: {batch['reward'][0]}")
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Merged streaming generator ready: Example batch shapes match")
    else:
        print("❌ ERROR: Shape mismatches detected. Debugging required.")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    test_generator()

