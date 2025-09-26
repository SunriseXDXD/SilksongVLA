#!/usr/bin/env python3
"""
Silksong VLA Training Script with SmolVLA

This script implements a Vision-Language-Action (VLA) agent for playing
Hollow Knight: Silksong using SmolVLA as the base model with hierarchical RL.
"""

import pyautogui
import numpy as np
import cv2
import time
import threading
from PIL import ImageGrab, Image
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from transformers import AutoModelForVision2Seq, AutoProcessor
from Controls import control_state, GetControlState, Cleanup, StopAllMovement
from config.screenconfig import ScreenConfig
from process_detector import ProcessDetector

# SmolVLA Configuration
class SmolVLAConfig:
    def __init__(self):
        self.model_name = "HuggingFaceM4/SmolVLM-Instruct"  # SmolVLA model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Fine-tuning parameters
        self.use_lora = True
        self.lora_rank = 16  # Smaller rank for SmolVLA
        self.learning_rate = 1e-4  # Lower learning rate for stability
        self.batch_size = 2  # Smaller batch size for memory efficiency
        self.grad_accumulation_steps = 8
        
        # Action space configuration for Silksong
        self.action_dim = 20  # Number of discrete actions
        self.action_discretization = 256  # SmolVLA uses 256-bin discretization
        
        # Training parameters
        self.max_episodes = 1000
        self.max_steps_per_episode = 1000
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Prompt templates for different game scenarios
        self.prompt_templates = {
            "combat": "In: What action should Hornet take to defeat the enemy? Out:",
            "exploration": "In: What action should Hornet take to explore the area? Out:",
            "platforming": "In: What action should Hornet take to navigate the platform? Out:",
            "general": "In: What action should Hornet take to progress in the game? Out:"
        }
        
        # Process detection settings
        self.require_game_focus = True  # Require game window to be focused
        self.game_wait_timeout = 60.0  # Seconds to wait for game to start
        self.game_check_interval = 2.0  # Seconds between game status checks
        self.auto_restart_training = True  # Automatically restart training when game starts

# SmolVLA-based RL Agent
class SmolVLAAgent:
    def __init__(self, config, screen_config):
        self.config = config
        self.screen_config = screen_config
        
        # Initialize SmolVLA model and processor
        print("Loading SmolVLA model...")
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            attn_implementation="flash_attention_1" if torch.cuda.is_available() else "eager",
            torch_dtype=config.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
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
        
        print("SmolVLA agent initialized successfully!")
    
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
            print("LoRA setup completed successfully!")
            
        except ImportError:
            print("PEFT not available, proceeding without LoRA")
            self.config.use_lora = False
    
    def _create_action_mapping(self):
        """Create mapping between SmolVLA actions and Silksong controls"""
        # Define discrete action space
        actions = [
            # Movement
            "MoveLeft", "MoveRight", "MoveUp", "MoveDown", "Dash", "Hook",
            # Combat
            "Attack", "DownSlash", "HoldAttack", 
            # Tools
            "UseUpTool", "UseMidTool", "UseDownTool",
            # Special
            "NoAction", "SpecialAttack"
        ]
        
        return {i: action for i, action in enumerate(actions)}
    
    def get_action_prompt(self, scenario="general"):
        """Get appropriate prompt for current game scenario"""
        return self.config.prompt_templates.get(scenario, self.config.prompt_templates["general"])
    
    def preprocess_image(self, image):
        """Preprocess image for SmolVLA input"""
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize and preprocess
        image = image.resize((224, 224))
        
        return image
    
    def select_action(self, image, scenario="general"):
        """Select action using SmolVLA with epsilon-greedy exploration"""
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get prompt
        prompt = self.get_action_prompt(scenario)
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=processed_image,
            return_tensors="pt"
        ).to(self.config.device)
        
        # Generate action
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract action from generated text
        action_id = self._extract_action_from_text(generated_text)
        
        return action_id
    
    def _extract_action_from_text(self, text):
        """Extract action ID from generated text"""
        # Simple keyword matching for action extraction
        action_keywords = {
            "left": 0, "right": 1, "up": 2, "down": 3,
            "attack": 4, "downslash": 5, "jumpattack": 6, "dashattack": 7,
            "jump": 8, "doublejump": 9, "dash": 10, "walljump": 11,
            "moveleftjump": 12, "moverightjump": 13, "moveleftdash": 14, "moverightdash": 15,
            "jumpdash": 16, "downslashdash": 17
        }
        
        text_lower = text.lower()
        
        for keyword, action_id in action_keywords.items():
            if keyword in text_lower:
                return action_id
        
        # Default action if no match found
        return random.randint(0, self.config.action_dim - 1)
    
    def execute_action(self, action_id, controls):
        """Execute the selected action"""
        action_name = self.action_mapping.get(action_id, "NoAction")
        
        # Reset all controls first
        StopAllMovement()
        
        # Execute action based on mapping
        if action_name == "MoveLeft":
            controls.MoveLeft = True
        elif action_name == "MoveRight":
            controls.MoveRight = True
        elif action_name == "MoveUp":
            controls.MoveUp = True
        elif action_name == "MoveDown":
            controls.MoveDown = True
        elif action_name == "Attack":
            controls.Attack = True
        elif action_name == "DownSlash":
            controls.DownSlash = True
        elif action_name == "JumpAttack":
            controls.JumpAttack = True
        elif action_name == "DashAttack":
            controls.DashAttack = True
        elif action_name == "Jump":
            controls.Jump = True
        elif action_name == "DoubleJump":
            controls.DoubleJump = True
        elif action_name == "Dash":
            controls.Dash = True
        elif action_name == "WallJump":
            controls.WallJump = True
        elif action_name == "MoveLeft+Jump":
            controls.MoveLeft = True
            controls.Jump = True
        elif action_name == "MoveRight+Jump":
            controls.MoveRight = True
            controls.Jump = True
        elif action_name == "MoveLeft+Dash":
            controls.MoveLeft = True
            controls.Dash = True
        elif action_name == "MoveRight+Dash":
            controls.MoveRight = True
            controls.Dash = True
        elif action_name == "Jump+Attack":
            controls.Jump = True
            controls.Attack = True
        elif action_name == "Dash+Attack":
            controls.Dash = True
            controls.Attack = True
        elif action_name == "Jump+Dash":
            controls.Jump = True
            controls.Dash = True
        elif action_name == "DownSlash+Dash":
            controls.DownSlash = True
            controls.Dash = True
        elif action_name == "SpecialAttack":
            controls.SpecialAttack = True
        
        # Apply controls
        GetControlState()
    
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
            return None
        
        # Sample batch
        batch = random.sample(self.experience_buffer, self.config.batch_size)
        
        # Prepare training inputs
        states = [exp['state'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        next_states = [exp['next_state'] for exp in batch]
        dones = [exp['done'] for exp in batch]
        
        # Convert to tensors
        states = torch.stack(states).to(self.config.device)
        actions = torch.tensor(actions).to(self.config.device)
        rewards = torch.tensor(rewards).to(self.config.device)
        next_states = torch.stack(next_states).to(self.config.device)
        dones = torch.tensor(dones).to(self.config.device)
        
        # Compute loss and update model
        # This is a simplified version - you'd need to implement proper RL loss
        loss = self._compute_rl_loss(states, actions, rewards, next_states, dones)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (self.step_count + 1) % self.config.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Optimizer step would go here
        
        return loss.item()
    
    def _compute_rl_loss(self, states, actions, rewards, next_states, dones):
        """Compute RL loss (simplified version)"""
        # This is a placeholder - implement proper RL algorithm (PPO, SAC, etc.)
        predicted_actions = self.model(states)
        loss = torch.nn.functional.cross_entropy(predicted_actions, actions)
        return loss
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
    
    def save_model(self, path):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"Model saved to {path}")

# Screen capture class
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
            self.capture_thread.join()
        print("Screen capture stopped")
    
    def _capture_loop(self):
        """Main capture loop"""
        while self.is_capturing:
            frame = self.capture_frame()
            with self.lock:
                self.frame_buffer.append(frame)
            time.sleep(0.033)  # ~30 FPS
    
    def capture_frame(self):
        """Capture a single frame"""
        try:
            # Capture screen
            screenshot = ImageGrab.grab()
            frame = np.array(screenshot)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply center crop
            frame = self._center_crop(frame)
            
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
    
    def _center_crop(self, frame):
        """Apply center crop"""
        h, w = frame.shape[:2]
        crop_h = min(h, self.config.height)
        crop_w = min(w, self.config.width)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return frame[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    def get_latest_frame(self):
        """Get the latest captured frame"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
            return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

# Main training class
class SilksongSmolVLATrainer:
    def __init__(self):
        self.screen_config = ScreenConfig()
        self.vla_config = SmolVLAConfig()
        
        # Initialize components
        self.screen_capture = ScreenCapture(self.screen_config)
        self.agent = SmolVLAAgent(self.vla_config, self.screen_config)
        
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
        
        self.FPS = 120
        
        # Frame stack for temporal information
        self.frame_stack = deque(maxlen=4)
        
        # HP tracking for damage-based rewards
        self.last_player_hp = None
        self.last_boss_hp = None
        
        # Survival bonus tracking
        self.survival_start_time = None
        self.last_survival_bonus_time = None
        self.survival_bonus_interval = 5.0  # 5 seconds
    
    def start_training(self):
        """Start the SmolVLA training process"""
        if self.is_training:
            print("Training already in progress")
            return
        
        self.is_training = True
        self.current_episode = 0
        self.current_step = 0
        
        # Start screen capture
        self.screen_capture.start_capture()
        
        # Start training thread
        training_thread = threading.Thread(target=self._training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        print("SmolVLA training started!")
    
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.screen_capture.stop_capture()
        print("Training stopped")
    
    def _initialize_frame_stack(self):
        """Initialize frame stack with current frames"""
        self.frame_stack.clear()
        for _ in range(4):
            frame = self.screen_capture.get_latest_frame()
            self.frame_stack.append(frame)
    
    def _check_game_status(self):
        """Check if game is still running and focused"""
        if not self.process_detector.is_silksong_running():
            print("Silksong process not found")
            return False
        
        if self.vla_config.require_game_focus:
            if not self.process_detector.is_game_focused():
                print("Game window not focused")
                return False
        
        return True
    
    def _handle_game_not_running(self):
        """Handle fallback behavior when game is not running"""
        print("Game not running or not focused. Waiting...")
        
        if self.vla_config.auto_restart_training:
            # Wait for game to start
            start_time = time.time()
            while time.time() - start_time < self.vla_config.game_wait_timeout:
                if self._check_game_status():
                    print("Game detected! Resuming training...")
                    return True
                time.sleep(self.vla_config.game_check_interval)
            
            print("Game wait timeout reached. Stopping training.")
            return False
        
        return False
    
    def _training_loop(self):
        """Main training loop"""
        print("Starting training loop...")
        
        # Initialize frame stack
        self._initialize_frame_stack()
        
        while self.is_training and self.current_episode < self.vla_config.max_episodes:
            # Check game status
            if not self._check_game_status():
                if not self._handle_game_not_running():
                    break
                continue
            
            # Start new episode
            self.current_episode += 1
            self.current_step = 0
            episode_reward = 0
            
            # Reset survival timer and HP tracking for new episode
            self.survival_start_time = None
            self.last_survival_bonus_time = None
            self.last_player_hp = None
            self.last_boss_hp = None
            
            print(f"Starting episode {self.current_episode}")
            
            # Episode loop
            while (self.is_training and 
                   self.current_step < self.vla_config.max_steps_per_episode):
                
                # Get current frame
                current_frame = self.screen_capture.get_latest_frame()
                
                # Determine scenario
                scenario = self._determine_scenario(current_frame)
                
                # Select action
                action_id = self.agent.select_action(current_frame, scenario)
                
                # Execute action
                self.agent.execute_action(action_id, self.controls)
                
                # Get reward
                reward = self._get_reward(True, current_frame)
                episode_reward += reward
                
                # Get next frame
                next_frame = self.screen_capture.get_latest_frame()
                
                # Store experience
                self.agent.store_experience(current_frame, action_id, reward, next_frame, False)
                
                # Training step
                loss = self.agent.train_step()
                if loss is not None:
                    self.agent.step_count += 1
                
                # Update exploration rate
                self.agent.update_epsilon()
                
                # Check if episode is done
                if self._is_episode_done():
                    break
                
                self.current_step += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(1 / self.FPS)  # 1 / FPS
            
            print(f"Episode {self.current_episode} completed. Reward: {episode_reward:.2f}")
            
            # Save model periodically
            if self.current_episode % 10 == 0:
                self.agent.save_model(f"smolvla_checkpoint_episode_{self.current_episode}")
    
    def _determine_scenario(self, frame):
        """Determine current game scenario based on frame content"""
        # Simple heuristic - in practice, you'd use more sophisticated detection
        # For now, default to general
        return "general"
    
    def _get_reward(self, success, frame):
        """Get reward for current action with damage tracking and survival bonuses"""
        reward = 0.0
        
        # Base reward for successful action
        if success:
            reward += 0.1
        else:
            reward -= 0.05
        
        # Get current game state to track damage
        current_state = GetControlState()
        
        # Track damage taken (negative reward)
        if hasattr(current_state, 'player_hp') and hasattr(self, 'last_player_hp'):
            damage_taken = self.last_player_hp - current_state.player_hp
            if damage_taken > 0:
                reward -= damage_taken  # Penalty for damage taken
                print(f"Damage taken: {damage_taken}, penalty: {-damage_taken}")
        
        # Track damage dealt (positive reward)
        if hasattr(current_state, 'boss_hp') and hasattr(self, 'last_boss_hp'):
            damage_dealt = self.last_boss_hp - current_state.boss_hp
            if damage_dealt > 0:
                reward += damage_dealt / 10.0  # Reward for damage dealt
                print(f"Damage dealt: {damage_dealt}, reward: {damage_dealt / 10.0}")
        
        # Update last known HP values
        if hasattr(current_state, 'player_hp'):
            self.last_player_hp = current_state.player_hp
        if hasattr(current_state, 'boss_hp'):
            self.last_boss_hp = current_state.boss_hp
        
        # Survival bonus: +0.1 for every 5 seconds survived
        current_time = time.time()
        if self.survival_start_time is None:
            self.survival_start_time = current_time
            self.last_survival_bonus_time = current_time
        
        # Check if it's time for survival bonus
        if current_time - self.last_survival_bonus_time >= self.survival_bonus_interval:
            reward += 0.1
            self.last_survival_bonus_time = current_time
            print(f"Survival bonus: +0.1 (survived {current_time - self.survival_start_time:.1f}s)")
        
        return reward
    
    def _is_episode_done(self):
        """Check if current episode should end"""
        # Simple termination condition
        return self.current_step >= self.vla_config.max_steps_per_episode

# Main execution
if __name__ == "__main__":
    trainer = SilksongSmolVLATrainer()
    
    try:
        print("Starting Silksong SmolVLA training...")
        print("Make sure Silksong is running and the game window is focused.")
        
        trainer.start_training()
        
        # Keep the main thread alive
        while trainer.is_training:
            time.sleep(1)
            
    except KeyboardInterrupt or pyautogui.keyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        trainer.stop_training()
        Cleanup()
        print("Training completed")
