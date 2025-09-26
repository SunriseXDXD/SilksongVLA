"""
Action Translator for Silksong RL Agent

This module parses macro action strings from VLM output and translates them
into executable action sequences or embeddings for RL policies.
"""

import re
import numpy as np
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

@dataclass
class ActionStep:
    """Represents a single step in a macro action sequence"""
    action: str
    duration: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"action": self.action}
        if self.duration is not None:
            result["duration"] = self.duration
        if self.parameters:
            result["parameters"] = self.parameters
        return result

class ActionTranslator:
    """
    Translates VLM macro strings into structured action sequences
    and generates embeddings for RL policies.
    """
    
    def __init__(self, 
                 embedding_model: Optional[str] = None,
                 device: str = "cpu"):
        self.device = device
        
        # Action vocabulary
        self.action_vocab = {
            "MoveLeft": {"type": "movement", "default_duration": 0.2},
            "MoveRight": {"type": "movement", "default_duration": 0.2},
            "MoveUp": {"type": "movement", "default_duration": 0.2},
            "MoveDown": {"type": "movement", "default_duration": 0.2},
            "Jump": {"type": "movement", "default_duration": 0.3},
            "DoubleJump": {"type": "movement", "default_duration": 0.5},
            "Dash": {"type": "movement", "default_duration": 0.1},
            "Attack": {"type": "combat", "default_duration": 0.2},
            "DownSlash": {"type": "combat", "default_duration": 0.3},
            "JumpAttack": {"type": "combat", "default_duration": 0.4},
            "DashAttack": {"type": "combat", "default_duration": 0.2},
            "UseTool": {"type": "ability", "default_duration": 0.3},
            "Hook": {"type": "ability", "default_duration": 0.4},
            "Bind": {"type": "ability", "default_duration": 0.5},
            "QuickMap": {"type": "utility", "default_duration": 0.5},
            "no_action": {"type": "utility", "default_duration": 0.1}
        }
        
        # Action patterns for parsing
        self.action_patterns = [
            r"(MoveLeft|MoveRight|MoveUp|MoveDown)",
            r"(Jump|DoubleJump)",
            r"(Dash)",
            r"(Attack|DownSlash|JumpAttack|DashAttack)",
            r"(UseTool|Hook|Bind|QuickMap)",
            r"(Wait|no_action)"
        ]
        
        # Duration patterns
        self.duration_pattern = r"\((\d+\.?\d*)s\)"
        
        # Initialize embedding model if specified
        self.embedding_model = None
        if embedding_model:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.embedding_model = self.embedding_model.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load embedding model {embedding_model}: {e}")
    
    def parse_macro_string(self, macro_string: str) -> List[ActionStep]:
        """
        Parse macro string into structured action sequence
        
        Examples:
        "JumpRight(0.3s) + Dash + Attack" -> [
            ActionStep("JumpRight", 0.3),
            ActionStep("Dash", 0.1),
            ActionStep("Attack", 0.2)
        ]
        """
        # Clean the input string
        macro_string = macro_string.strip().replace("+", ",").replace("and", ",")
        
        # Split into individual actions
        action_parts = [part.strip() for part in macro_string.split(",") if part.strip()]
        
        action_sequence = []
        
        for part in action_parts:
            # Extract duration if present
            duration_match = re.search(self.duration_pattern, part)
            duration = None
            
            if duration_match:
                duration = float(duration_match.group(1))
                # Remove duration from action name
                action_name = re.sub(self.duration_pattern, "", part).strip()
            else:
                action_name = part.strip()
            
            # Map to known actions
            matched_action = self._match_action(action_name)
            
            if matched_action:
                # Use default duration if not specified
                if duration is None:
                    duration = self.action_vocab[matched_action]["default_duration"]
                
                action_step = ActionStep(
                    action=matched_action,
                    duration=duration
                )
                
                action_sequence.append(action_step)
        
        return action_sequence
    
    def _match_action(self, action_name: str) -> Optional[str]:
        """Match action name to known vocabulary"""
        action_name_lower = action_name.lower()
        
        # Direct match
        if action_name in self.action_vocab:
            return action_name
        
        # Case-insensitive match
        for action in self.action_vocab:
            if action.lower() == action_name_lower:
                return action
        
        # Pattern matching
        for pattern in self.action_patterns:
            match = re.search(pattern, action_name, re.IGNORECASE)
            if match:
                matched = match.group(1)
                # Find the canonical action name
                for action in self.action_vocab:
                    if action.lower() == matched.lower():
                        return action
        
        # Compound action matching
        if "jump" in action_name_lower and "right" in action_name_lower:
            return "MoveRight"  # Simplified: jump right = move right + jump
        
        if "jump" in action_name_lower and "left" in action_name_lower:
            return "MoveLeft"   # Simplified: jump left = move left + jump
        
        if "dash" in action_name_lower and "attack" in action_name_lower:
            return "DashAttack"
        
        if "jump" in action_name_lower and "attack" in action_name_lower:
            return "JumpAttack"
        
        # Default fallback
        return "no_action"
    
    def action_sequence_to_embeddings(self, 
                                    action_sequence: List[ActionStep],
                                    subgoal: str) -> torch.Tensor:
        """
        Convert action sequence and subgoal to embeddings for RL policy
        
        Args:
            action_sequence: List of ActionStep objects
            subgoal: Text description of the subgoal
            
        Returns:
            Combined embedding tensor
        """
        if self.embedding_model is None:
            # Fallback: simple one-hot encoding
            return self._simple_action_encoding(action_sequence)
        
        # Encode subgoal
        subgoal_embedding = self.embedding_model.encode(
            subgoal, 
            convert_to_tensor=True,
            device=self.device
        )
        
        # Encode action sequence
        action_descriptions = []
        for step in action_sequence:
            desc = f"{step.action}"
            if step.duration:
                desc += f" for {step.duration}s"
            action_descriptions.append(desc)
        
        action_text = " -> ".join(action_descriptions)
        action_embedding = self.embedding_model.encode(
            action_text,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Combine embeddings
        combined_embedding = torch.cat([subgoal_embedding, action_embedding])
        
        return combined_embedding
    
    def _simple_action_encoding(self, action_sequence: List[ActionStep]) -> torch.Tensor:
        """Simple one-hot encoding fallback"""
        action_list = list(self.action_vocab.keys())
        encoding_dim = len(action_list)
        
        # Create sequence encoding
        sequence_length = min(len(action_sequence), 5)  # Max 5 actions
        encoding = torch.zeros(sequence_length, encoding_dim)
        
        for i, step in enumerate(action_sequence[:sequence_length]):
            if step.action in action_list:
                action_idx = action_list.index(step.action)
                encoding[i, action_idx] = 1.0
        
        return encoding.flatten()
    
    def create_action_plan(self, 
                          subgoal: str, 
                          macro: str) -> Dict[str, Any]:
        """
        Create complete action plan from VLM output
        
        Args:
            subgoal: Text description of the subgoal
            macro: Macro action string
            
        Returns:
            Complete action plan dictionary
        """
        # Parse macro string
        action_sequence = self.parse_macro_string(macro)
        
        # Generate embeddings
        if self.embedding_model:
            embeddings = self.action_sequence_to_embeddings(action_sequence, subgoal)
        else:
            embeddings = None
        
        # Calculate total duration
        total_duration = sum(step.duration for step in action_sequence if step.duration)
        
        # Create action plan
        action_plan = {
            "subgoal": subgoal,
            "macro": macro,
            "action_sequence": [step.to_dict() for step in action_sequence],
            "total_duration": total_duration,
            "num_actions": len(action_sequence),
            "embeddings": embeddings
        }
        
        return action_plan
    
    def execute_action_sequence(self, 
                              action_sequence: List[ActionStep],
                              control_interface) -> bool:
        """
        Execute action sequence using control interface
        
        Args:
            action_sequence: List of ActionStep objects
            control_interface: Game control interface
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            for step in action_sequence:
                if step.action == "no_action":
                    if step.duration:
                        time.sleep(step.duration)
                    continue
                
                # Get control method
                control_method = getattr(control_interface, step.action, None)
                
                if control_method is None:
                    print(f"Warning: Control method {step.action} not found")
                    continue
                
                # Execute action
                if step.duration:
                    # For actions with duration, start and stop
                    if hasattr(control_interface, f"Start{step.action}"):
                        start_method = getattr(control_interface, f"Start{step.action}")
                        stop_method = getattr(control_interface, f"Stop{step.action}")
                        
                        start_method()
                        time.sleep(step.duration)
                        stop_method()
                    else:
                        # For simple actions, just call the method
                        control_method()
                        if step.duration > 0.1:
                            time.sleep(step.duration - 0.1)
                else:
                    # For instant actions
                    control_method()
                
                # Small delay between actions
                time.sleep(0.05)
            
            return True
            
        except Exception as e:
            print(f"Error executing action sequence: {e}")
            return False

class ActionEmbeddingNet(nn.Module):
    """
    Neural network for learning action embeddings from subgoals and action sequences
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 max_sequence_length: int = 10):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        
        # Action embedding layer
        self.action_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Subgoal processing (simplified - in practice you'd use a text encoder)
        self.subgoal_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Sequence processing
        self.sequence_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, 
                action_indices: torch.Tensor,
                subgoal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            action_indices: Tensor of action indices (batch_size, seq_len)
            subgoal_embedding: Subgoal embedding (batch_size, embedding_dim)
            
        Returns:
            Combined action embedding
        """
        # Embed actions
        action_embeds = self.action_embedding(action_indices)
        
        # Process sequence
        lstm_out, _ = self.sequence_lstm(action_embeds)
        
        # Pool sequence (use last hidden state)
        sequence_embedding = lstm_out[:, -1, :]
        
        # Process subgoal
        subgoal_proj = self.subgoal_projection(subgoal_embedding)
        
        # Combine
        combined = sequence_embedding + subgoal_proj
        output = self.output_projection(combined)
        
        return output

def test_action_translator():
    """Test the action translator"""
    print("Testing Action Translator...")
    
    translator = ActionTranslator()
    
    # Test macro parsing
    test_macros = [
        "JumpRight(0.3s) + Dash + Attack",
        "MoveLeft + Jump + DashAttack",
        "Avoid projectile and counterattack",
        "Dash + JumpAttack + DownSlash",
        "MoveRight(0.5s) + Attack + Wait(0.2s)"
    ]
    
    for macro in test_macros:
        print(f"\nMacro: {macro}")
        sequence = translator.parse_macro(macro)
        print(f"Parsed sequence: {[step.to_dict() for step in sequence]}")
    
    # Test action plan creation
    subgoal = "Avoid projectile and counterattack from right side"
    macro = "JumpRight(0.3s) + Dash + Attack"
    
    action_plan = translator.create_action_plan(subgoal, macro)
    print(f"\nAction Plan: {action_plan}")
    
    return translator

if __name__ == "__main__":
    test_action_translator()
