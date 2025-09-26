import numpy as np
import cv2
import pyautogui
import time
import threading
from PIL import ImageGrab
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from Controls import control_state, GetControlState, Cleanup, StopAllMovement
from screenconfig import ScreenConfig

# Screen capture class
class ScreenCapture:
    def __init__(self, config):
        self.config = config
        self.is_capturing = False
        self.capture_thread = None
        self.frame_buffer = deque(maxlen=2)  # Buffer for latest frames
        self.lock = threading.Lock()
        
    def start_capture(self):
        """Start continuous screen capture"""
        if not self.is_capturing:
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print("Screen capture started")
    
    def stop_capture(self):
        """Stop screen capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        print("Screen capture stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_capturing:
            frame = self.capture_frame()
            if frame is not None:
                with self.lock:
                    self.frame_buffer.append(frame)
            time.sleep(self.config.frame_delay)
    
    def capture_frame(self):
        """Capture a single frame from the fullscreen game window"""
        try:
            # Capture fullscreen
            screenshot = ImageGrab.grab(
                bbox=(
                    self.config.capture_x,
                    self.config.capture_y,
                    self.config.capture_x + self.config.capture_width,
                    self.config.capture_y + self.config.capture_height
                )
            )
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert from RGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Apply center crop if enabled
            if hasattr(self.config, 'center_crop') and self.config.center_crop:
                frame = self._center_crop(frame)
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def _center_crop(self, frame):
        """Apply center crop to frame"""
        height, width = frame.shape[:2]
        crop_width = self.config.crop_width
        crop_height = self.config.crop_height
        
        # Calculate crop coordinates
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height
        
        # Ensure crop coordinates are within bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(width, end_x)
        end_y = min(height, end_y)
        
        return frame[start_y:end_y, start_x:end_x]
    
    def get_latest_frame(self):
        """Get the latest captured frame"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
            return None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for RL input"""
        if frame is None:
            return None
            
        try:
            # Resize frame
            resized = cv2.resize(frame, (self.config.target_width, self.config.target_height))
            
            # Convert to grayscale if specified
            if self.config.grayscale:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                # Add channel dimension
                resized = np.expand_dims(resized, axis=-1)
            
            # Normalize pixel values
            if self.config.normalize:
                resized = resized.astype(np.float32) / 255.0
            
            return resized
            
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None

# Action space for Silksong
class ActionSpace:
    def __init__(self):
        # Define available actions
        self.actions = [
            'no_op',
            'move_left',
            'move_right', 
            'move_up',
            'move_down',
            'jump',
            'jump_short',
            'jump_long',
            'double_jump',
            'attack',
            'down_slash',
            'dash',
            'dash_attack',
            'jump_attack',
            'use_tool',
            'use_up_tool',
            'use_down_tool',
            'hook',
            'bind',
            'quick_map'
        ]
        
        self.action_map = {
            0: self._no_op,
            1: self._move_left,
            2: self._move_right,
            3: self._move_up,
            4: self._move_down,
            5: self._jump,
            6: self._jump_short,
            7: self._jump_long,
            8: self._double_jump,
            9: self._attack,
            10: self._down_slash,
            11: self._dash,
            12: self._dash_attack,
            13: self._jump_attack,
            14: self._use_tool,
            15: self._use_up_tool,
            16: self._use_down_tool,
            17: self._hook,
            18: self._bind,
            19: self._quick_map
        }
        
        self.n_actions = len(self.actions)
    
    def execute_action(self, action_id, controls):
        """Execute the specified action"""
        if action_id in self.action_map:
            return self.action_map[action_id](controls)
        return False
    
    def _no_op(self, controls):
        """No operation"""
        return True
    
    def _move_left(self, controls):
        """Move left"""
        controls.MoveLeft()
        time.sleep(0.1)
        controls.StopMoveLeft()
        return True
    
    def _move_right(self, controls):
        """Move right"""
        controls.MoveRight()
        time.sleep(0.1)
        controls.StopMoveRight()
        return True
    
    def _move_up(self, controls):
        """Move up"""
        controls.MoveUp()
        time.sleep(0.1)
        controls.StopMoveUp()
        return True
    
    def _move_down(self, controls):
        """Move down"""
        controls.MoveDown()
        time.sleep(0.1)
        controls.StopMoveDown()
        return True
    
    def _jump(self, controls):
        """Normal jump"""
        controls.StartJump()
        time.sleep(0.2)
        controls.StopJump()
        return True
    
    def _jump_short(self, controls):
        """Short jump"""
        controls.StartJump()
        time.sleep(0.1)
        controls.StopJump()
        return True
    
    def _jump_long(self, controls):
        """Long jump"""
        controls.StartJump()
        time.sleep(0.4)
        controls.StopJump()
        return True
    
    def _double_jump(self, controls):
        """Double jump"""
        controls.StartJump()
        time.sleep(0.2)
        controls.DoubleJump()
        time.sleep(0.2)
        controls.StopJump()
        return True
    
    def _attack(self, controls):
        """Attack"""
        controls.Attack()
        return True
    
    def _dash(self, controls):
        """Dash"""
        controls.Dash()
        return True
    
    def _dash_attack(self, controls):
        """Dash attack"""
        controls.DashAttack()
        return True
    
    def _jump_attack(self, controls):
        """Jump attack"""
        controls.JumpAttack()
        return True
    
    def _use_tool(self, controls):
        """Use tool"""
        controls.UseMidTool()
        return True
    
    def _use_up_tool(self, controls):
        """Use up tool"""
        controls.UseUpTool()
        return True
    
    def _use_down_tool(self, controls):
        """Use down tool"""
        controls.UseDownTool()
        return True
    
    def _hook(self, controls):
        """Hook"""
        controls.Hook()
        return True
    
    def _bind(self, controls):
        """Bind ability"""
        controls.Bind()
        return True
    
    def _quick_map(self, controls):
        """Quick map"""
        controls.QuickMap(duration=0.5)
        return True

# Simple CNN for RL
class DQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Main training class
class SilksongTrainer:
    def __init__(self):
        self.config = ScreenConfig()
        self.screen_capture = ScreenCapture(self.config)
        self.action_space = ActionSpace()
        
        # Import controls
        import Controls
        self.controls = Controls
        
        # RL parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 4  # Frame stacking
        self.action_size = self.action_space.n_actions
        
        # Initialize DQN
        self.policy_net = DQN(self.input_channels, self.action_size).to(self.device)
        self.target_net = DQN(self.input_channels, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(10000)
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        self.target_update = 1000
        
        # Frame stacking
        self.frame_stack = deque(maxlen=self.input_channels)
        
        # Training state
        self.is_training = False
        self.episode_count = 0
        self.step_count = 0
        
    def start_training(self):
        """Start the training process"""
        print("Starting Silksong RL training...")
        self.is_training = True
        
        # Start screen capture
        self.screen_capture.start_capture()
        
        # Wait a moment for capture to initialize
        time.sleep(1.0)
        
        # Initialize frame stack
        self._initialize_frame_stack()
        
        # Start training loop
        self._training_loop()
    
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.screen_capture.stop_capture()
        self.controls.Cleanup()
        print("Training stopped")
    
    def _initialize_frame_stack(self):
        """Initialize the frame stack with initial frames"""
        while len(self.frame_stack) < self.input_channels:
            frame = self.screen_capture.get_latest_frame()
            if frame is not None:
                processed_frame = self.screen_capture.preprocess_frame(frame)
                if processed_frame is not None:
                    self.frame_stack.append(processed_frame)
            time.sleep(0.1)
    
    def _get_state(self):
        """Get current state from frame stack"""
        if len(self.frame_stack) == self.input_channels:
            # Stack frames
            state = np.stack(list(self.frame_stack), axis=0)
            # Convert to tensor
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return state
        return None
    
    def _training_loop(self):
        """Main training loop"""
        while self.is_training:
            try:
                # Get current state
                state = self._get_state()
                if state is None:
                    print("Waiting for valid state...")
                    time.sleep(0.1)
                    continue
                
                # Select action
                action = self._select_action(state)
                
                # Execute action
                reward = self._execute_action(action)
                
                # Get next state
                next_frame = self.screen_capture.get_latest_frame()
                if next_frame is not None:
                    processed_next_frame = self.screen_capture.preprocess_frame(next_frame)
                    if processed_next_frame is not None:
                        self.frame_stack.append(processed_next_frame)
                        
                        next_state = self._get_state()
                        done = self._check_episode_done()
                        
                        # Store experience
                        self.memory.push(state, action, reward, next_state, done)
                        
                        # Learn
                        self._learn()
                        
                        # Update frame stack
                        state = next_state
                
                self.step_count += 1
                
                # Update target network
                if self.step_count % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Target network updated at step {self.step_count}")
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.03)
                
            except KeyboardInterrupt:
                print("Training interrupted by user")
                break
            except Exception as e:
                print(f"Error in training loop: {e}")
                time.sleep(1.0)
    
    def _select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
    
    def _execute_action(self, action):
        """Execute action and return reward"""
        try:
            success = self.action_space.execute_action(action, self.controls)
            if success:
                # Simple reward function - can be customized
                return 0.1  # Small positive reward for successful action
            else:
                return -0.1  # Small negative reward for failed action
        except Exception as e:
            print(f"Error executing action {action}: {e}")
            return -0.5
    
    def _check_episode_done(self):
        """Check if episode is done"""
        # Simple episode termination - can be customized
        # For now, just use step count as episode length
        return self.step_count % 1000 == 0
    
    def _learn(self):
        """Train the model"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1})
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))
        
        # Compute loss
        loss = nn.MSELoss()(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.step_count % 100 == 0:
            print(f"Step {self.step_count}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")

# Main execution
if __name__ == "__main__":
    trainer = SilksongTrainer()
    
    try:
        trainer.start_training()
    except KeyboardInterrupt:
        print("Training stopped by user")
    finally:
        trainer.stop_training()
