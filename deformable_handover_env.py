"""
Step 2: Environment Setup - DeformableHandoverEnv

Custom Gymnasium environment for deformable object handover with PyBullet simulation.
Falls back to simplified physics if PyBullet is unavailable.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional
import warnings

# Try to import PyBullet, fall back to mock if unavailable
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    warnings.warn("PyBullet not available. Using simplified physics simulation.")

from data_preparation import merged_generator


class SimplifiedPhysicsSimulator:
    """Simplified physics simulator as fallback when PyBullet is unavailable."""
    
    def __init__(self, human_distance=None):
        self.timestep = 1.0 / 240.0
        self.gravity = -9.81
        self.human_distance_override = human_distance
        
        # Object states
        self.arm_position = np.array([0.0, 0.0, 0.5])
        self.arm_velocity = np.zeros(3)
        self.towel_position = np.array([0.3, 0.0, 0.5])
        self.towel_velocity = np.zeros(3)
        self.towel_mass = 0.2
        self.towel_stiffness = 200.0  # Increased stiffness to prevent dropping
        
        # Human avatar (position set by reset)
        self.human_position = np.array([0.5, 0.0, 0.5])
        self.contact_force = 0.0
        
        # Grasp state
        self.grasped = True  # Start with object grasped to avoid immediate drop
        self.grasp_distance = 0.15  # Increased grasp range
        
    def reset(self):
        """Reset simulation state."""
        self.arm_position = np.array([0.0, 0.0, 0.5])
        self.arm_velocity = np.zeros(3)
        self.towel_position = np.array([0.0, 0.0, 0.5])  # Start at arm position
        self.towel_velocity = np.zeros(3)
        
        # Set human position based on curriculum (if specified)
        if self.human_distance_override is not None:
            # Fixed distance for curriculum
            distance = self.human_distance_override
        else:
            # Random distance between 0.5-1.0m
            distance = np.random.uniform(0.5, 1.0)
        
        # Place human at specified distance in front of arm
        self.human_position = np.array([distance, 0.0, 0.5]) + np.random.uniform(-0.1, 0.1, 3)
        self.contact_force = 0.0
        self.grasped = True  # Start grasped
        
    def step_simulation(self, substeps: int = 1):
        """Advance physics simulation."""
        for _ in range(substeps):
            # Check grasp first
            dist_to_arm = np.linalg.norm(self.towel_position - self.arm_position)
            if dist_to_arm < self.grasp_distance:
                self.grasped = True
            
            # Apply physics based on grasp state
            if not self.grasped:
                # Apply gravity with damping
                self.towel_velocity[2] += self.gravity * self.timestep * 0.5  # Reduced gravity effect
                self.towel_velocity *= 0.95  # Air resistance
                self.towel_position += self.towel_velocity * self.timestep
                # Ground collision
                if self.towel_position[2] < 0.1:
                    self.towel_position[2] = 0.1
                    self.towel_velocity[2] = 0
            else:
                # Towel follows arm when grasped with some springiness
                spring_force = (self.arm_position - self.towel_position) * self.towel_stiffness
                self.towel_velocity += spring_force * self.timestep / self.towel_mass
                self.towel_velocity *= 0.8  # Damping
                self.towel_position += self.towel_velocity * self.timestep
            
            # Simulate human pull (if close to human)
            dist_to_human = np.linalg.norm(self.towel_position - self.human_position)
            if dist_to_human < 0.2 and self.grasped:
                # Human is trying to take object
                pull_direction = (self.human_position - self.towel_position) / (dist_to_human + 1e-6)
                pull_force = 8.0 * (0.2 - dist_to_human) / 0.2
                self.contact_force = pull_force
                
                # Human actually pulls the object
                self.towel_velocity += pull_direction * pull_force * self.timestep / self.towel_mass
            else:
                self.contact_force = 0.0
    
    def apply_action(self, action: np.ndarray):
        """Apply action to robot arm."""
        # Action is 6DoF: position delta (3) + orientation (3)
        position_delta = action[:3] * 0.05  # Scale to reasonable range
        self.arm_velocity = position_delta / self.timestep
        self.arm_position += position_delta
        
        # Clamp arm position to workspace
        self.arm_position = np.clip(self.arm_position, [-0.5, -0.5, 0.1], [0.5, 0.5, 1.0])
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            'arm_position': self.arm_position.copy(),
            'towel_position': self.towel_position.copy(),
            'human_position': self.human_position.copy(),
            'contact_force': self.contact_force,
            'grasped': self.grasped
        }


class PyBulletSimulator:
    """PyBullet-based physics simulator."""
    
    def __init__(self, gui: bool = False):
        self.gui = gui
        self.client_id = None
        self.arm_id = None
        self.towel_id = None
        self.human_id = None
        self.timestep = 1.0 / 240.0
        
    def reset(self):
        """Initialize or reset PyBullet simulation."""
        # Disconnect if already connected
        if self.client_id is not None:
            p.disconnect(self.client_id)
        
        # Connect to PyBullet
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.timestep, physicsClientId=self.client_id)
        
        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # Load robot arm (using Kuka as placeholder)
        self.arm_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client_id
        )
        
        # Create deformable towel (simplified as soft body)
        # Since soft body creation is complex, use a multi-link rigid approximation
        self.towel_id = self._create_deformable_towel()
        
        # Create human avatar (kinematic body)
        self.human_id = p.loadURDF(
            "sphere2.urdf",
            basePosition=[0.5 + np.random.uniform(-0.1, 0.1), 
                         0.0 + np.random.uniform(-0.1, 0.1),
                         0.5 + np.random.uniform(-0.1, 0.1)],
            physicsClientId=self.client_id
        )
        p.changeDynamics(self.human_id, -1, mass=0, physicsClientId=self.client_id)
        
    def _create_deformable_towel(self) -> int:
        """Create a deformable towel approximation."""
        # Create a thin box as towel (simplified)
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.01],
            physicsClientId=self.client_id
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.01],
            rgbaColor=[0.8, 0.2, 0.2, 1.0],
            physicsClientId=self.client_id
        )
        
        towel_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.3, 0.0, 0.5],
            physicsClientId=self.client_id
        )
        
        # Set deformable-like properties
        p.changeDynamics(
            towel_id, -1,
            linearDamping=0.4,
            angularDamping=0.4,
            restitution=0.1,
            lateralFriction=1.0,
            physicsClientId=self.client_id
        )
        
        return towel_id
    
    def step_simulation(self, substeps: int = 1):
        """Advance physics simulation."""
        for _ in range(substeps):
            p.stepSimulation(physicsClientId=self.client_id)
    
    def apply_action(self, action: np.ndarray):
        """Apply action to robot arm."""
        # Apply joint torques (simplified: position control)
        num_joints = p.getNumJoints(self.arm_id, physicsClientId=self.client_id)
        for i in range(min(6, num_joints)):
            target_position = action[i] * 2.0  # Scale action
            p.setJointMotorControl2(
                self.arm_id, i,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=100,
                physicsClientId=self.client_id
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        # Get arm end-effector position
        arm_state = p.getLinkState(self.arm_id, p.getNumJoints(self.arm_id, self.client_id) - 1,
                                   physicsClientId=self.client_id)
        arm_pos = np.array(arm_state[0])
        
        # Get towel position
        towel_pos, _ = p.getBasePositionAndOrientation(self.towel_id, physicsClientId=self.client_id)
        towel_pos = np.array(towel_pos)
        
        # Get human position
        human_pos, _ = p.getBasePositionAndOrientation(self.human_id, physicsClientId=self.client_id)
        human_pos = np.array(human_pos)
        
        # Estimate contact force
        contact_points = p.getContactPoints(self.towel_id, self.human_id, physicsClientId=self.client_id)
        contact_force = sum([pt[9] for pt in contact_points]) if contact_points else 0.0
        
        # Check if grasped
        grasp_contacts = p.getContactPoints(self.arm_id, self.towel_id, physicsClientId=self.client_id)
        grasped = len(grasp_contacts) > 0
        
        return {
            'arm_position': arm_pos,
            'towel_position': towel_pos,
            'human_position': human_pos,
            'contact_force': contact_force,
            'grasped': grasped
        }
    
    def close(self):
        """Clean up PyBullet connection."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None


class DeformableHandoverEnv(gym.Env):
    """
    Custom Gymnasium environment for deformable object handover.
    
    Observation space:
        - image: (84, 84, 3) RGB image
        - effort: (6,) joint efforts/torques
        - imu: (6,) IMU data
        - audio: (16000,) 1-second audio waveform
    
    Action space:
        - (6,) normalized 6DoF end-effector control
    
    Reward:
        +10 for successful handover (effort change > 5N)
        -1 per step
        -50 for drop (object z < 0.1m) or excessive force (> 50N)
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, use_gui: bool = False, noise_std: float = 0.05, adversarial_mode: bool = False, 
                 human_distance: float = None):
        """
        Initialize the environment.
        
        Args:
            use_gui: Whether to show PyBullet GUI (if available)
            noise_std: Standard deviation of observation noise
            adversarial_mode: If True, adds random ±0.2m perturbations to human poses
            human_distance: Override human distance for curriculum learning (default: random 0.5-1.0m)
        """
        super().__init__()
        
        self.use_gui = use_gui
        self.noise_std = noise_std
        self.adversarial_mode = adversarial_mode
        self.human_distance_override = human_distance
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),
            'effort': spaces.Box(-100, 100, (6,), dtype=np.float32),
            'imu': spaces.Box(-10, 10, (6,), dtype=np.float32),
            'audio': spaces.Box(0, 1, (16000,), dtype=np.float32)
        })
        
        # Define action space: 6DoF end-effector control (normalized)
        self.action_space = spaces.Box(-1, 1, (6,), dtype=np.float32)
        
        # Initialize data generator
        self.data_generator = None
        self.current_batch = None
        self.batch_index = 0
        
        # Initialize physics simulator
        if PYBULLET_AVAILABLE:
            print("Using PyBullet physics simulator")
            self.simulator = PyBulletSimulator(gui=use_gui)
        else:
            print("Using simplified physics simulator")
            self.simulator = SimplifiedPhysicsSimulator(human_distance=self.human_distance_override)
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 70  # Reduced from 100 to force faster action
        self.previous_effort = np.zeros(6)
        self.handover_achieved = False
        
    def _get_next_observation_from_stream(self) -> Dict[str, np.ndarray]:
        """Get next observation from the merged data generator."""
        # Initialize generator if needed
        if self.data_generator is None:
            self.data_generator = merged_generator(batch_size=32)
        
        # Get new batch if needed
        if self.current_batch is None or self.batch_index >= 32:
            try:
                self.current_batch = next(self.data_generator)
                self.batch_index = 0
            except StopIteration:
                # Reinitialize generator if exhausted
                self.data_generator = merged_generator(batch_size=32)
                self.current_batch = next(self.data_generator)
                self.batch_index = 0
        
        # Extract observation from current batch
        obs = {
            'image': self.current_batch['observation']['image'][self.batch_index],
            'effort': self.current_batch['observation']['effort'][self.batch_index].astype(np.float32),
            'imu': self.current_batch['observation']['imu'][self.batch_index].astype(np.float32),
            'audio': self.current_batch['observation']['audio'][self.batch_index].astype(np.float32)
        }
        
        # Resize audio to 16000 samples (1 second at 16kHz)
        audio = obs['audio']
        if len(audio) < 16000:
            # Pad with zeros if too short
            obs['audio'] = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
        elif len(audio) > 16000:
            # Truncate if too long
            obs['audio'] = audio[:16000]
        
        # Normalize audio to [0, 1]
        audio = obs['audio']
        audio_min, audio_max = audio.min(), audio.max()
        if audio_max > audio_min:
            obs['audio'] = (audio - audio_min) / (audio_max - audio_min)
        else:
            obs['audio'] = np.zeros(16000, dtype=np.float32)
        
        self.batch_index += 1
        
        return obs
    
    def _add_noise(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to observations for realism."""
        noisy_obs = observation.copy()
        
        # Add noise to effort
        noisy_obs['effort'] = observation['effort'] + np.random.normal(
            0, self.noise_std * 10, size=observation['effort'].shape
        ).astype(np.float32)
        noisy_obs['effort'] = np.clip(noisy_obs['effort'], -100, 100)
        
        # Add noise to IMU
        noisy_obs['imu'] = observation['imu'] + np.random.normal(
            0, self.noise_std * 1, size=observation['imu'].shape
        ).astype(np.float32)
        noisy_obs['imu'] = np.clip(noisy_obs['imu'], -10, 10)
        
        # Add small noise to audio
        noisy_obs['audio'] = observation['audio'] + np.random.normal(
            0, self.noise_std * 0.1, size=observation['audio'].shape
        ).astype(np.float32)
        noisy_obs['audio'] = np.clip(noisy_obs['audio'], 0, 1)
        
        return noisy_obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulator
        self.simulator.reset()
        
        # Apply adversarial perturbations if enabled
        if self.adversarial_mode and hasattr(self.simulator, 'human_position'):
            perturbation = np.random.uniform(-0.2, 0.2, size=3)
            self.simulator.human_position += perturbation
        
        # Get initial observation from data stream
        observation = self._get_next_observation_from_stream()
        
        # Add noise
        observation = self._add_noise(observation)
        
        # Reset episode state
        self.step_count = 0
        self.previous_effort = observation['effort'].copy()
        self.handover_achieved = False
        self.prev_arm_to_human_dist = None  # Reset for progress reward
        
        info = {
            'step': self.step_count,
            'handover_achieved': self.handover_achieved
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 6DoF end-effector control (normalized to [-1, 1])
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self.step_count += 1
        
        # Denormalize action (scale to appropriate range)
        action = np.clip(action, -1, 1)
        
        # Apply action to simulator
        self.simulator.apply_action(action)
        
        # Simulate physics (30 Hz = 1/30 second per step)
        substeps = int((1.0 / 30.0) / self.simulator.timestep)
        self.simulator.step_simulation(substeps=substeps)
        
        # Apply adversarial perturbations during episode (10% chance per step)
        if self.adversarial_mode and np.random.rand() < 0.1:
            if hasattr(self.simulator, 'human_position'):
                perturbation = np.random.uniform(-0.2, 0.2, size=3)
                self.simulator.human_position += perturbation
        
        # Get simulation state
        sim_state = self.simulator.get_state()
        
        # Get new observation from stream
        observation = self._get_next_observation_from_stream()
        
        # Update effort based on simulation (blend with streamed data)
        effort_change = sim_state['contact_force']
        observation['effort'] = observation['effort'] + effort_change
        
        # Add noise
        observation = self._add_noise(observation)
        
        # Compute reward with dense shaping
        reward = 0.0
        terminated = False
        failure_reason = None
        
        # Get current state
        current_effort_magnitude = np.linalg.norm(observation['effort'])
        previous_effort_magnitude = np.linalg.norm(self.previous_effort)
        effort_delta = abs(current_effort_magnitude - previous_effort_magnitude)
        towel_z = sim_state['towel_position'][2]
        grasped = sim_state.get('grasped', False)
        
        # SPARSE reward shaping - focus ONLY on task completion
        
        # 1. PROGRESS REWARD ONLY (no passive bonuses)
        arm_to_human_dist = np.linalg.norm(sim_state['arm_position'] - sim_state['human_position'])
        
        # Track previous distance for progress reward
        if self.prev_arm_to_human_dist is None:
            self.prev_arm_to_human_dist = arm_to_human_dist
        
        # Strong reward for getting closer
        progress = self.prev_arm_to_human_dist - arm_to_human_dist
        reward += 5.0 * progress  # Doubled from 2.5 - ONLY way to get positive rewards
        
        # Update for next step
        self.prev_arm_to_human_dist = arm_to_human_dist
        
        # 2. CONTACT REWARD (human touching towel)
        contact_force = sim_state['contact_force']
        if contact_force > 0:
            reward += 1.0  # Good signal of handover attempt
        if contact_force > 2.0:
            reward += 2.0  # Strong contact very good
        
        # Terminal conditions
        
        # SUCCESS: Handover achieved - HUGE bonus
        if effort_delta > 5.0 and grasped and contact_force > 2.0:
            reward += 150.0  # Increased from 50 - make success VERY valuable!
            self.handover_achieved = True
            terminated = True
        
        # FAILURE: Drop
        if towel_z < 0.1:
            reward -= 20.0  # Moderate penalty
            terminated = True
            failure_reason = 'drop'
        
        # FAILURE: Excessive force
        if current_effort_magnitude > 60.0:
            reward -= 20.0  # Moderate penalty
            terminated = True
            failure_reason = 'excessive_force'
        
        # Small constant time penalty (no passive rewards!)
        reward -= 0.1  # Constant penalty to encourage finishing fast
        
        # Update previous effort
        self.previous_effort = observation['effort'].copy()
        
        # Check for truncation (max steps)
        truncated = self.step_count >= self.max_steps
        
        info = {
            'step': self.step_count,
            'handover_achieved': self.handover_achieved,
            'effort_delta': effort_delta,
            'towel_z': towel_z,
            'contact_force': contact_force,
            'grasped': grasped,
            'effort_magnitude': current_effort_magnitude,
            'arm_to_human_dist': arm_to_human_dist,
            'failure_reason': failure_reason,
            'terminated': terminated,
            'truncated': truncated
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (returns RGB array)."""
        if hasattr(self, 'current_batch') and self.current_batch is not None:
            return self.current_batch['observation']['image'][self.batch_index - 1]
        return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources."""
        if hasattr(self.simulator, 'close'):
            self.simulator.close()


# Register environment with Gymnasium
try:
    gym.register(
        id='DeformableHandover-v0',
        entry_point='deformable_handover_env:DeformableHandoverEnv',
        max_episode_steps=100,
    )
    print("✓ Environment registered: DeformableHandover-v0")
except gym.error.Error:
    # Already registered
    pass

