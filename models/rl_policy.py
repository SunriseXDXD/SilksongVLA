"""
RL Policy for Silksong Agent

This module implements reinforcement learning policies that can use VLM-generated
subgoals and embeddings to make action decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random

class GoalConditionedPolicy(nn.Module):
    """
    Goal-conditioned policy network that takes both state and goal embeddings
    as input to produce action probabilities.
    """
    
    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim + goal_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                state: torch.Tensor, 
                goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, state_dim)
            goal: Goal embedding tensor (batch_size, goal_dim)
            
        Returns:
            Action logits (batch_size, action_dim)
        """
        # Concatenate state and goal
        combined = torch.cat([state, goal], dim=-1)
        
        # Forward through network
        logits = self.network(combined)
        
        return logits
    
    def get_action(self, 
                   state: torch.Tensor, 
                   goal: torch.Tensor,
                   epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: State tensor
            goal: Goal embedding tensor
            epsilon: Exploration rate
            
        Returns:
            Tuple of (action, action_probabilities)
        """
        with torch.no_grad():
            logits = self.forward(state, goal)
            probs = F.softmax(logits, dim=-1)
            
            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = torch.multinomial(probs, num_samples=1).item()
            
            return action, probs

class GoalConditionedCritic(nn.Module):
    """
    Goal-conditioned critic network for estimating state-action values.
    """
    
    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim + goal_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                state: torch.Tensor, 
                goal: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, state_dim)
            goal: Goal embedding tensor (batch_size, goal_dim)
            action: Action tensor (batch_size, action_dim)
            
        Returns:
            Q-value (batch_size, 1)
        """
        # One-hot encode action if needed
        if action.dim() == 1:
            action = F.one_hot(action, num_classes=self.action_dim).float()
        
        # Concatenate state, goal, and action
        combined = torch.cat([state, goal, action], dim=-1)
        
        # Forward through network
        q_value = self.network(combined)
        
        return q_value

class GoalConditionedAgent:
    """
    Goal-conditioned RL agent that uses VLM-generated goal embeddings
    to guide action selection.
    """
    
    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 256):
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy and critic networks
        self.policy = GoalConditionedPolicy(state_dim, goal_dim, action_dim).to(self.device)
        self.critic = GoalConditionedCritic(state_dim, goal_dim, action_dim).to(self.device)
        self.target_critic = GoalConditionedCritic(state_dim, goal_dim, action_dim).to(self.device)
        
        # Copy weights to target
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Training state
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        print(f"Goal-conditioned agent initialized on {self.device}")
    
    def store_experience(self, 
                        state: np.ndarray,
                        goal: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        next_goal: np.ndarray,
                        done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': state,
            'goal': goal,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_goal': next_goal,
            'done': done
        }
        self.buffer.append(experience)
    
    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample batch from replay buffer"""
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = random.sample(self.buffer, self.batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        goals = torch.FloatTensor([exp['goal'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        next_goals = torch.FloatTensor([exp['next_goal'] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
        
        return {
            'states': states,
            'goals': goals,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'next_goals': next_goals,
            'dones': dones
        }
    
    def update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update critic network"""
        states = batch['states']
        goals = batch['goals']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        next_goals = batch['next_goals']
        dones = batch['dones']
        
        with torch.no_grad():
            # Get next actions from policy
            next_logits = self.policy(next_states, next_goals)
            next_probs = F.softmax(next_logits, dim=-1)
            next_actions = torch.argmax(next_probs, dim=-1)
            
            # Get target Q-values
            target_q_values = self.target_critic(next_states, next_goals, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values.squeeze()
        
        # Get current Q-values
        current_q_values = self.critic(states, goals, actions).squeeze()
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_policy(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update policy network"""
        states = batch['states']
        goals = batch['goals']
        
        # Get policy actions
        logits = self.policy(states, goals)
        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        
        # Get Q-values for policy actions
        q_values = self.critic(states, goals, actions).squeeze()
        
        # Compute policy loss (maximize Q-values)
        policy_loss = -q_values.mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def update_target_network(self):
        """Update target critic network"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.tau) + param.data * self.tau
            )
    
    def train_step(self) -> Tuple[float, float]:
        """Perform one training step"""
        batch = self.sample_batch()
        if batch is None:
            return 0.0, 0.0
        
        # Update critic
        critic_loss = self.update_critic(batch)
        
        # Update policy
        policy_loss = self.update_policy(batch)
        
        # Update target network
        self.update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return critic_loss, policy_loss
    
    def get_action(self, 
                   state: np.ndarray, 
                   goal: np.ndarray) -> Tuple[int, float]:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        
        action, probs = self.policy.get_action(state_tensor, goal_tensor, self.epsilon)
        action_prob = probs[0, action].item()
        
        return action, action_prob
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        
        print(f"Model loaded from {path}")

class HierarchicalPolicy:
    """
    Hierarchical policy that combines VLM-generated macro actions
    with RL-refined primitive actions.
    """
    
    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 macro_action_dim: int = 20):
        
        # High-level policy (chooses macro actions)
        self.high_level_policy = GoalConditionedPolicy(
            state_dim, goal_dim, macro_action_dim
        )
        
        # Low-level policy (refines macro actions)
        self.low_level_policy = GoalConditionedPolicy(
            state_dim + macro_action_dim, goal_dim, action_dim
        )
        
        self.macro_action_dim = macro_action_dim
        self.action_dim = action_dim
        
    def get_action(self, 
                   state: np.ndarray,
                   goal: np.ndarray,
                   use_refinement: bool = True) -> Tuple[int, int]:
        """
        Get action using hierarchical policy
        
        Args:
            state: Current state
            goal: Goal embedding
            use_refinement: Whether to use low-level refinement
            
        Returns:
            Tuple of (macro_action, primitive_action)
        """
        # Get macro action
        macro_action, _ = self.high_level_policy.get_action(
            torch.FloatTensor(state).unsqueeze(0),
            torch.FloatTensor(goal).unsqueeze(0)
        )
        
        if use_refinement:
            # Create macro action embedding
            macro_embedding = F.one_hot(
                torch.tensor(macro_action), 
                num_classes=self.macro_action_dim
            ).float()
            
            # Combine state and macro embedding
            combined_state = np.concatenate([state, macro_embedding.numpy()])
            
            # Get refined primitive action
            primitive_action, _ = self.low_level_policy.get_action(
                torch.FloatTensor(combined_state).unsqueeze(0),
                torch.FloatTensor(goal).unsqueeze(0)
            )
            
            return macro_action, primitive_action
        else:
            # Direct mapping from macro to primitive
            return macro_action, self._macro_to_primitive(macro_action)
    
    def _macro_to_primitive(self, macro_action: int) -> int:
        """Map macro action to primitive action (simplified)"""
        # This is a simplified mapping - in practice, you'd have
        # a more sophisticated mapping or sequence execution
        mapping = {
            0: 0,  # no_action -> no_action
            1: 1,  # move_left -> move_left
            2: 2,  # move_right -> move_right
            3: 3,  # jump -> jump
            4: 4,  # dash -> dash
            5: 5,  # attack -> attack
            # ... etc.
        }
        return mapping.get(macro_action, 0)

def test_rl_policy():
    """Test the RL policy components"""
    print("Testing RL Policy...")
    
    # Create agent
    state_dim = 64  # Example state dimension
    goal_dim = 128  # Example goal embedding dimension
    action_dim = 20  # Example action space size
    
    agent = GoalConditionedAgent(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim
    )
    
    # Test forward pass
    state = np.random.randn(state_dim)
    goal = np.random.randn(goal_dim)
    
    action, prob = agent.get_action(state, goal)
    print(f"Action: {action}, Probability: {prob:.4f}")
    
    # Test training step
    # Fill buffer with random experiences
    for _ in range(1000):
        exp_state = np.random.randn(state_dim)
        exp_goal = np.random.randn(goal_dim)
        exp_action = random.randint(0, action_dim - 1)
        exp_reward = random.random()
        exp_next_state = np.random.randn(state_dim)
        exp_next_goal = np.random.randn(goal_dim)
        exp_done = random.random() < 0.1
        
        agent.store_experience(
            exp_state, exp_goal, exp_action, exp_reward,
            exp_next_state, exp_next_goal, exp_done
        )
    
    # Perform training step
    critic_loss, policy_loss = agent.train_step()
    print(f"Training - Critic Loss: {critic_loss:.4f}, Policy Loss: {policy_loss:.4f}")
    
    return agent

if __name__ == "__main__":
    test_rl_policy()
