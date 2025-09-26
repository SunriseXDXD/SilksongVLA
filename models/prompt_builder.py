"""
Prompt Builder for Silksong RL Agent

This module constructs structured prompts for Vision-Language Models
by serializing object state and game context into natural language.
"""

from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass

@dataclass
class GameContext:
    """Context information about the current game state"""
    episode: int
    step: int
    total_steps: int
    score: Optional[float] = None
    time_remaining: Optional[float] = None
    difficulty: str = "normal"
    player_status: str = "active"

class PromptBuilder:
    """
    Builds structured prompts for VLMs by converting game state to natural language.
    """
    
    def __init__(self, 
                 include_detailed_physics: bool = True,
                 include_tactical_context: bool = True,
                 include_memory_context: bool = False):
        self.include_detailed_physics = include_detailed_physics
        self.include_tactical_context = include_tactical_context
        self.include_memory_context = include_memory_context
        
        # Templates for different prompt components
        self.system_prompt = """You are controlling Hornet in Hollow Knight: Silksong. 
You are an expert player who understands the game mechanics, enemy patterns, and optimal strategies.
Your task is to analyze the current game state and decide on the best action plan.

Respond with:
Subgoal: <brief description of immediate objective>
Macro: <sequence of actions from [MoveLeft, MoveRight, Jump, Dash, Attack, DownSlash, JumpAttack, DashAttack]>

Keep your response concise and actionable."""
        
        self.action_descriptions = {
            "MoveLeft": "Move left horizontally",
            "MoveRight": "Move right horizontally", 
            "MoveUp": "Move up (climb/jump)",
            "MoveDown": "Move down (drop/fall)",
            "Jump": "Jump upward",
            "Dash": "Quick dash in current direction",
            "Attack": "Basic nail attack",
            "DownSlash": "Downward slash attack",
            "JumpAttack": "Attack while jumping",
            "DashAttack": "Attack while dashing",
            "UseTool": "Use equipped tool",
            "Hook": "Use hook ability",
            "Bind": "Use bind ability"
        }
    
    def serialize_object(self, obj: Dict[str, Any], obj_type: str) -> str:
        """Serialize a single object to readable text"""
        parts = [f"{obj_type.capitalize()}"]
        
        # Position and velocity
        if 'x' in obj and 'y' in obj:
            parts.append(f"at ({obj['x']}, {obj['y']})")
        
        if self.include_detailed_physics and 'vel_x' in obj and 'vel_y' in obj:
            parts.append(f"moving ({obj['vel_x']:.1f}, {obj['vel_y']:.1f})")
        
        # Game-specific properties
        if 'hp' in obj:
            parts.append(f"HP: {obj['hp']}")
        
        if 'phase' in obj:
            parts.append(f"Phase: {obj['phase']}")
        
        if 'damage' in obj:
            parts.append(f"Damage: {obj['damage']}")
        
        return " ".join(parts)
    
    def serialize_game_state(self, game_state: Dict[str, Any]) -> str:
        """Convert game state dictionary to readable text"""
        lines = ["Current game state:"]
        
        # Player state
        if 'player' in game_state['objects']:
            player = game_state['objects']['player'][0]  # Assume single player
            lines.append(f"- Player: {self.serialize_object(player, 'player')}")
        
        # Enemy states
        if 'boss' in game_state['objects']:
            for i, boss in enumerate(game_state['objects']['boss']):
                lines.append(f"- Boss {i+1}: {self.serialize_object(boss, 'boss')}")
        
        # Projectiles
        if 'projectiles' in game_state['objects']:
            projectiles = game_state['objects']['projectiles']
            if projectiles:
                lines.append(f"- Projectiles: {len(projectiles)} active")
                for i, proj in enumerate(projectiles[:3]):  # Show first 3
                    lines.append(f"  * {self.serialize_object(proj, 'projectile')}")
                if len(projectiles) > 3:
                    lines.append(f"  * ... and {len(projectiles) - 3} more")
        
        # Hazards
        if 'hazards' in game_state['objects']:
            hazards = game_state['objects']['hazards']
            if hazards:
                lines.append(f"- Hazards: {len(hazards)} present")
        
        # Platforms (if detected)
        if 'platforms' in game_state['objects']:
            platforms = game_state['objects']['platforms']
            if platforms:
                lines.append(f"- Platforms: {len(platforms)} detected")
        
        return "\n".join(lines)
    
    def add_tactical_context(self, game_state: Dict[str, Any]) -> List[str]:
        """Add tactical analysis based on game state"""
        context_lines = []
        
        if not self.include_tactical_context:
            return context_lines
        
        player = game_state['objects'].get('player', [{}])[0]
        bosses = game_state['objects'].get('boss', [])
        projectiles = game_state['objects'].get('projectile', [])
        
        # Player positioning analysis
        if 'x' in player and 'y' in player:
            player_x, player_y = player['x'], player['y']
            
            # Check if player is in danger zone
            for proj in projectiles:
                if 'x' in proj and 'y' in proj:
                    dist = ((player_x - proj['x'])**2 + (player_y - proj['y'])**2)**0.5
                    if dist < 30:
                        context_lines.append("WARNING: Player in immediate projectile danger!")
                        break
            
            # Position relative to bosses
            for boss in bosses:
                if 'x' in boss and 'y' in boss:
                    boss_x, boss_y = boss['x'], boss['y']
                    dist = ((player_x - boss_x)**2 + (player_y - boss_y)**2)**0.5
                    
                    if dist < 50:
                        context_lines.append("Player very close to boss")
                    elif dist > 150:
                        context_lines.append("Player far from boss")
                    
                    # Height advantage analysis
                    if player_y < boss_y - 20:
                        context_lines.append("Player has height advantage")
                    elif player_y > boss_y + 20:
                        context_lines.append("Player at height disadvantage")
        
        # Boss phase analysis
        for boss in bosses:
            if 'phase' in boss and 'hp' in boss:
                phase, hp = boss['phase'], boss['hp']
                if phase == 1 and hp < 90:
                    context_lines.append("Boss approaching phase transition")
                elif phase == 2:
                    context_lines.append("Boss in aggressive phase 2")
        
        return context_lines
    
    def build_prompt(self, 
                    game_state: Dict[str, Any], 
                    context: Optional[GameContext] = None,
                    memory: Optional[List[str]] = None) -> str:
        """
        Build complete prompt for VLM
        """
        prompt_parts = [self.system_prompt]
        
        # Add game state
        prompt_parts.append(self.serialize_game_state(game_state))
        
        # Add tactical context
        tactical_context = self.add_tactical_context(game_state)
        if tactical_context:
            prompt_parts.append("\nTactical analysis:")
            for line in tactical_context:
                prompt_parts.append(f"- {line}")
        
        # Add game context
        if context:
            prompt_parts.append(f"\nGame context:")
            prompt_parts.append(f"- Episode: {context.episode}, Step: {context.step}")
            if context.score is not None:
                prompt_parts.append(f"- Score: {context.score}")
            if context.time_remaining is not None:
                prompt_parts.append(f"- Time remaining: {context.time_remaining:.1f}s")
            prompt_parts.append(f"- Player status: {context.player_status}")
        
        # Add memory context
        if self.include_memory_context and memory:
            prompt_parts.append("\nRecent actions and outcomes:")
            for mem in memory[-3:]:  # Last 3 memories
                prompt_parts.append(f"- {mem}")
        
        # Add action instruction
        prompt_parts.append("\nDecide the next action plan.")
        prompt_parts.append("Output:")
        prompt_parts.append("Subgoal: <short description of immediate objective>")
        prompt_parts.append("Macro: <sequence of actions>")
        
        return "\n".join(prompt_parts)
    
    def build_simple_prompt(self, game_state: Dict[str, Any]) -> str:
        """Build a simplified prompt for faster processing"""
        prompt = "You are Hornet in Silksong. Game state:\n"
        
        # Simplified state
        if 'player' in game_state['objects']:
            player = game_state['objects']['player'][0]
            prompt += f"Player at ({player.get('x', 0)}, {player.get('y', 0)})\n"
        
        if 'boss' in game_state['objects']:
            boss = game_state['objects']['boss'][0]
            prompt += f"Boss at ({boss.get('x', 0)}, {boss.get('y', 0)}), HP: {boss.get('hp', 0)}\n"
        
        projectiles = game_state['objects'].get('projectile', [])
        if projectiles:
            prompt += f"{len(projectiles)} projectiles\n"
        
        prompt += "\nSubgoal: <objective>\nMacro: <MoveLeft/MoveRight/Jump/Dash/Attack/DownSlash>"
        
        return prompt
    
    def parse_vlm_response(self, response: str) -> Dict[str, str]:
        """Parse VLM response to extract subgoal and macro"""
        lines = response.strip().split('\n')
        result = {"subgoal": "", "macro": ""}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Subgoal:"):
                result["subgoal"] = line.replace("Subgoal:", "").strip()
            elif line.startswith("Macro:"):
                result["macro"] = line.replace("Macro:", "").strip()
        
        return result

def test_prompt_builder():
    """Test the prompt builder"""
    from models.object_extractor import create_mock_game_state
    
    # Create mock game state
    game_state = {
        "objects": create_mock_game_state(),
        "frame_count": 100
    }
    
    # Create context
    context = GameContext(
        episode=1,
        step=250,
        total_steps=1000,
        score=1500,
        time_remaining=45.0
    )
    
    # Build prompt
    builder = PromptBuilder()
    prompt = builder.build_prompt(game_state, context)
    
    print("Generated Prompt:")
    print("=" * 50)
    print(prompt)
    print("=" * 50)
    
    # Test simple prompt
    simple_prompt = builder.build_simple_prompt(game_state)
    print("\nSimple Prompt:")
    print("-" * 30)
    print(simple_prompt)
    
    return prompt

if __name__ == "__main__":
    test_prompt_builder()
