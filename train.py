import numpy as np
import cv2
import time
import threading
from PIL import ImageGrab,Image
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from transformers import AutoModelForVision2Seq, AutoProcessor
from Controls import control_state, GetControlState, Cleanup, StopAllMovement
from screenconfig import ScreenConfig
from process_detector import ProcessDetector

# OpenVLA Configuration
class OpenVLAConfig:
    def __init__(self):
        self.model_name = "openvla/openvla-7b"  # Pretrained OpenVLA model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Fine-tuning parameters
        self.use_lora = True
        self.lora_rank = 32
        self.learning_rate = 5e-4
        self.batch_size = 4  # Reduced for memory efficiency
        self.grad_accumulation_steps = 4
        
        # Action space configuration for Silksong
        self.action_dim = 20  # Number of discrete actions
        self.action_discretization = 256  # OpenVLA uses 256-bin discretization
        
        # Training parameters
        self.max_episodes = 1000
        self.max_steps_per_episode = 1000
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Prompt templates for different game scenarios
        self.prompt_templates = {
            "combat": "In: What action should the character take to defeat the enemy?",
            "exploration": "In: What action should the character take to explore the area?",
            "platforming": "In: What action should the character take to navigate the platform?",
            "general": "In: What action should the character take to progress in the game?"
        }
        
        # Process detection settings
        self.require_game_focus = True  # Require game window to be focused
        self.game_wait_timeout = 60.0  # Seconds to wait for game to start
        self.game_check_interval = 2.0  # Seconds between game status checks
        self.auto_restart_training = True  # Automatically restart training when game starts

# OpenVLA-based RL Agent
class OpenVLAAgent:
    def __init__(self, config, screen_config):
        self.config = config
        self.screen_config = screen_config
        
        # Initialize OpenVLA model and processor
        print("Loading OpenVLA model...")
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(config.device)
        
        # Setup LoRA if enabled
        if config.use_lora:
            self._setup_lora()
        
        # Action space mapping
        self.action_mapping = self._create_action_mapping()
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.episode_count = 0
        self.step_count = 0
        
        print("OpenVLA agent initialized successfully!")
    
    def _setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print("LoRA setup completed!")
            
        except ImportError:
            print("PEFT not available, continuing without LoRA")
            self.config.use_lora = False
    
    def _create_action_mapping(self):
        """Create mapping between OpenVLA actions and Silksong controls"""
        return {
            0: "no_action",
            1: "move_left", 
            2: "move_right",
            3: "move_up",
            4: "move_down",
            5: "jump_short",
            6: "jump_medium",
            7: "jump_long",
            8: "double_jump",
            9: "attack",
            10: "down_slash",
            11: "dash",
            12: "dash_attack",
            13: "jump_attack",
            14: "use_tool",
            15: "use_up_tool",
            16: "use_down_tool",
            17: "hook",
            18: "bind",
            19: "quick_map"
        }
    
    def get_action_prompt(self, scenario="general"):
        """Get appropriate prompt for current game scenario"""
        base_prompt = self.config.prompt_templates.get(scenario, self.config.prompt_templates["general"])
        return f"{base_prompt}\nOut:"
    
    def preprocess_image(self, image):
        """Preprocess image for OpenVLA input"""
        if image is None:
            return None
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        return pil_image
    
    def select_action(self, image, scenario="general"):
        """Select action using OpenVLA with epsilon-greedy exploration"""
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, len(self.action_mapping) - 1)
        
        try:
            # Preprocess image
            pil_image = self.preprocess_image(image)
            if pil_image is None:
                return random.randint(0, len(self.action_mapping) - 1)
            
            # Get prompt
            prompt = self.get_action_prompt(scenario)
            
            # Prepare inputs
            inputs = self.processor(prompt, pil_image).to(
                self.config.device, 
                dtype=self.config.torch_dtype
            )
            
            # Get action prediction from OpenVLA
            with torch.no_grad():
                # OpenVLA predicts continuous actions, we'll discretize
                action_logits = self.model.predict_action(
                    **inputs, 
                    unnorm_key=None,  # We'll handle normalization ourselves
                    do_sample=False
                )
                
                # Convert continuous action to discrete
                if isinstance(action_logits, torch.Tensor):
                    action_logits = action_logits.cpu().numpy()
                
                # Simple discretization: map to action space
                action_id = int(action_logits[0] * len(self.action_mapping)) % len(self.action_mapping)
                
                return action_id
                
        except Exception as e:
            print(f"Error in action selection: {e}")
            return random.randint(0, len(self.action_mapping) - 1)
    
    def execute_action(self, action_id, controls):
        """Execute the selected action"""
        if action_id not in self.action_mapping:
            return False
        
        action_name = self.action_mapping[action_id]
        
        try:
            if action_name == "no_action":
                return True
            elif action_name == "move_left":
                controls.MoveLeft()
                time.sleep(0.1)
                controls.StopMoveLeft()
            elif action_name == "move_right":
                controls.MoveRight()
                time.sleep(0.1)
                controls.StopMoveRight()
            elif action_name == "move_up":
                controls.MoveUp()
                time.sleep(0.1)
                controls.StopMoveUp()
            elif action_name == "move_down":
                controls.MoveDown()
                time.sleep(0.1)
                controls.StopMoveDown()
            elif action_name == "jump_short":
                controls.StartJump()
                time.sleep(0.1)
                controls.StopJump()
            elif action_name == "jump_medium":
                controls.StartJump()
                time.sleep(0.2)
                controls.StopJump()
            elif action_name == "jump_long":
                controls.StartJump()
                time.sleep(0.4)
                controls.StopJump()
            elif action_name == "double_jump":
                controls.StartJump()
                time.sleep(0.2)
                controls.DoubleJump()
                time.sleep(0.2)
                controls.StopJump()
            elif action_name == "attack":
                controls.Attack()
            elif action_name == "down_slash":
                controls.DownSlash()
            elif action_name == "dash":
                controls.Dash()
            elif action_name == "dash_attack":
                controls.DashAttack()
            elif action_name == "jump_attack":
                controls.JumpAttack()
            elif action_name == "use_tool":
                controls.UseMidTool()
            elif action_name == "use_up_tool":
                controls.UseUpTool()
            elif action_name == "use_down_tool":
                controls.UseDownTool()
            elif action_name == "hook":
                controls.Hook()
            elif action_name == "bind":
                controls.Bind()
            elif action_name == "quick_map":
                controls.QuickMap(duration=0.5)
            
            return True
            
        except Exception as e:
            print(f"Error executing action {action_name}: {e}")
            return False
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience for training"""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def train_step(self):
        """Perform one training step"""
        if len(self.experience_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        batch = random.sample(list(self.experience_buffer), self.config.batch_size)
        
        # Prepare training data
        states = [exp['state'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        
        # Simple supervised learning: train OpenVLA to predict actions
        try:
            self.model.train()
            
            total_loss = 0
            for i, (state, action) in enumerate(zip(states, actions)):
                if state is None:
                    continue
                
                pil_image = self.preprocess_image(state)
                if pil_image is None:
                    continue
                
                # Create training prompt
                prompt = self.get_action_prompt("general")
                
                # Prepare inputs
                inputs = self.processor(prompt, pil_image).to(
                    self.config.device, 
                    dtype=self.config.torch_dtype
                )
                
                # Create target action (convert to continuous representation)
                target_action = torch.tensor([action / len(self.action_mapping)], 
                                           device=self.config.device, 
                                           dtype=self.config.torch_dtype)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Simple loss: MSE between predicted and target action
                if hasattr(outputs, 'logits'):
                    # Use logits if available
                    predicted_action = outputs.logits.mean(dim=-1)  # Simplified
                    loss = nn.MSELoss()(predicted_action, target_action)
                else:
                    # Fallback loss
                    loss = torch.tensor(0.1, device=self.config.device, requires_grad=True)
                
                # Backward pass
                loss = loss / self.config.grad_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
            
            # Update weights
            if (self.step_count + 1) % self.config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Note: You'd need to set up an optimizer for proper training
                # optimizer.step()
                # optimizer.zero_grad()
            
            avg_loss = total_loss / len(batch)
            
            if self.step_count % 100 == 0:
                print(f"Step {self.step_count}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
                
        except Exception as e:
            print(f"Error in training step: {e}")
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
    
    def save_model(self, path):
        """Save the fine-tuned model"""
        if self.config.use_lora:
            self.model.save_pretrained(path)
        else:
            print("Warning: Full model saving not implemented without LoRA")

# Screen capture class (same as before but adapted for OpenVLA)
class ScreenCapture:
    def __init__(self, config):
        self.config = config
        self.is_capturing = False
        self.capture_thread = None
        self.frame_buffer = deque(maxlen=2)
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
        """Main capture loop"""
        while self.is_capturing:
            frame = self.capture_frame()
            if frame is not None:
                with self.lock:
                    self.frame_buffer.append(frame)
            time.sleep(self.config.frame_delay)
    
    def capture_frame(self):
        """Capture a single frame"""
        try:
            screenshot = ImageGrab.grab(
                bbox=(
                    self.config.capture_x,
                    self.config.capture_y,
                    self.config.capture_x + self.config.capture_width,
                    self.config.capture_y + self.config.capture_height
                )
            )
            
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Apply center crop if enabled
            if hasattr(self.config, 'center_crop') and self.config.center_crop:
                frame = self._center_crop(frame)
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def _center_crop(self, frame):
        """Apply center crop"""
        height, width = frame.shape[:2]
        crop_width = getattr(self.config, 'crop_width', 1920)
        crop_height = getattr(self.config, 'crop_height', 1080)
        
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height) // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height
        
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

# Main training class
class SilksongOpenVLATrainer:
    def __init__(self):
        self.screen_config = ScreenConfig()
        self.vla_config = OpenVLAConfig()
        
        # Initialize components
        self.screen_capture = ScreenCapture(self.screen_config)
        self.agent = OpenVLAAgent(self.vla_config, self.screen_config)
        
        # Initialize process detector
        self.process_detector = ProcessDetector()
        self.process_detector.set_focus_requirement(self.vla_config.require_game_focus)
        
        # Import controls
        import Controls
        self.controls = Controls
        
        # Training state
        self.is_training = False
        self.current_episode = 0
        self.current_step = 0
        self.game_window_handle = None
        
        # Frame stack for temporal information
        self.frame_stack = deque(maxlen=4)
    
    def start_training(self):
        """Start the OpenVLA training process"""
        print("Starting Silksong OpenVLA training...")
        
        # Check if Silksong is running
        print("Checking if Silksong is running...")
        is_ready, window_handle = self.process_detector.is_game_ready()
        
        if not is_ready:
            print(f"Silksong is not ready. Waiting for game to start (timeout: {self.vla_config.game_wait_timeout}s)...")
            is_ready, window_handle = self.process_detector.wait_for_game(self.vla_config.game_wait_timeout)
            
            if not is_ready:
                print("ERROR: Silksong is not running or not focused. Please start the game and try again.")
                print("Game info:")
                info = self.process_detector.get_game_info()
                for key, value in info.items():
                    print(f"  {key}: {value}")
                return False
        
        print("Silksong is ready! Starting training...")
        self.game_window_handle = window_handle
        self.is_training = True
        
        # Start screen capture
        self.screen_capture.start_capture()
        
        # Wait for capture to initialize
        time.sleep(1.0)
        
        # Initialize frame stack
        self._initialize_frame_stack()
        
        # Start training loop
        self._training_loop()
    
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.screen_capture.stop_capture()
        self.controls.Cleanup()
        
        # Save model
        self.agent.save_model("silksong_openvla_finetuned")
        print("Training stopped and model saved")

    def _initialize_frame_stack(self):
        """Initialize frame stack with current frames"""
        for _ in range(4):
            frame = self.screen_capture.get_latest_frame()
            if frame is not None:
                self.frame_stack.append(frame)
            time.sleep(0.1)

    def _check_game_status(self):
        """Check if game is still running and focused"""
        is_ready, window_handle = self.process_detector.is_game_ready()
        
        if not is_ready:
            print("WARNING: Silksong is no longer ready (not running or not focused)")
            
            if self.vla_config.auto_restart_training:
                print("Attempting to handle game disconnection...")
                return self._handle_game_not_running()
            else:
                print("Stopping training...")
                self.is_training = False
                return False
        
        return True
    
    def _handle_game_not_running(self):
        """Handle fallback behavior when game is not running"""
        print("Game is not running. Executing fallback behavior...")
        
        # Stop all current actions
        self.controls.StopAllMovement()
        
        # Wait and periodically check if game comes back
        wait_time = 0
        max_wait_time = self.vla_config.game_wait_timeout
        
        while wait_time < max_wait_time and self.is_training:
            print(f"Waiting for game... ({wait_time:.1f}s / {max_wait_time}s)")
            
            # Check if game is ready
            is_ready, window_handle = self.process_detector.is_game_ready()
            if is_ready:
                print("Game is back! Resuming training.")
                self.game_window_handle = window_handle
                return True
            
            # Sleep for a short interval
            time.sleep(2.0)
            wait_time += 2.0
        
        print("Game did not return within timeout. Stopping training.")
        self.is_training = False
        return False

    def _training_loop(self):
        """Main training loop"""
        while self.is_training and self.current_episode < self.vla_config.max_episodes:
            try:
                print(f"Starting Episode {self.current_episode + 1}")

                # Reset episode state
                episode_reward = 0
                self.current_step = 0
                
                # Initialize frame stack for new episode
                self.frame_stack.clear()
                self._initialize_frame_stack()
                
                while (self.is_training and 
                       self.current_step < self.vla_config.max_steps_per_episode):
                    
                    # Check game status periodically
                    if self.current_step % int(self.vla_config.game_check_interval / self.vla_config.action_delay) == 0:
                        if not self._check_game_status():
                            break
                    
                    # Get current state
                    current_frame = self.screen_capture.get_latest_frame()
                    if current_frame is None:
                        time.sleep(0.1)
                        continue
                    
                    # Determine scenario (simple heuristic based on frame content)
                    scenario = self._determine_scenario(current_frame)
                    
                    # Select action using OpenVLA
                    action_id = self.agent.select_action(current_frame, scenario)
                    
                    # Execute action
                    success = self.agent.execute_action(action_id, self.controls)
                    
                    # Get reward
                    reward = self._get_reward(success, current_frame)
                    episode_reward += reward
                    
                    # Get next state
                    next_frame = self.screen_capture.get_latest_frame()
                    
                    # Check if episode is done
                    done = self._is_episode_done()
                    
                    # Store experience
                    self.agent.store_experience(
                        current_frame, action_id, reward, next_frame, done
                    )
                    
                    # Train agent
                    self.agent.train_step()
                    
                    # Update counters
                    self.current_step += 1
                    self.agent.step_count += 1
                    
                    # Update exploration rate
                    self.agent.update_epsilon()
                    
                    # Small delay
                    time.sleep(0.05)
                
                # Episode complete
                print(f"Episode {self.current_episode + 1} completed. "
                      f"Reward: {episode_reward:.2f}, Steps: {self.current_step}")
                
                self.current_episode += 1
                
                # Reset game state
                self.controls.ResetJumpState()
                time.sleep(1.0)  # Brief pause between episodes
                
            except KeyboardInterrupt:
                print("Training interrupted by user")
                break
            except Exception as e:
                print(f"Error in training loop: {e}")
                time.sleep(1.0)
    
    def _determine_scenario(self, frame):
        """Determine current game scenario based on frame content"""
        # Simple heuristic - can be improved with computer vision
        # For now, return general scenario
        return "general"
    
    def _get_reward(self, success, frame):
        """Get reward for current action"""
        if not success:
            return -0.5
        
        # Simple reward function
        base_reward = 0.1
        
        # Can add more sophisticated reward shaping here
        # For example: detect enemies, platforms, collectibles, etc.
        
        return base_reward
    
    def _is_episode_done(self):
        """Check if current episode should end"""
        # Simple episode termination conditions
        if self.current_step >= self.vla_config.max_steps_per_episode:
            return True
        
        # Can add more conditions like:
        # - Character death
        # - Level completion
        # - Timeout
        
        return False

# Main execution
if __name__ == "__main__":
    trainer = SilksongOpenVLATrainer()
    
    try:
        trainer.start_training()
    except KeyboardInterrupt:
        print("Training stopped by user")
    finally:
        trainer.stop_training()
