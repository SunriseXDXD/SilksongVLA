#!/usr/bin/env python3
"""
Comprehensive test script for Silksong VLA model components

This script tests each model component in isolation to ensure they work correctly
before integration into the full training pipeline.
"""

import sys
import os
import numpy as np
import cv2
import torch
from PIL import Image
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.object_extractor import SimpleObjectExtractor, GameObject, create_mock_game_state
from models.prompt_builder import PromptBuilder, GameContext
from models.vlm_interface import MockVLM, VLMInterface
from models.action_translator import ActionTranslator, ActionStep
from models.rl_policy import GoalConditionedAgent, GoalConditionedPolicy

def test_object_extractor():
    """Test the object extractor component"""
    print("\n=== Testing Object Extractor ===")
    
    try:
        # Create extractor
        extractor = SimpleObjectExtractor()
        print("✓ Object extractor created successfully")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        print("✓ Test frame created")
        
        # Test preprocessing
        processed = extractor.preprocess_frame(test_frame)
        assert processed.shape == (64, 64, 3), f"Expected (64, 64, 3), got {processed.shape}"
        print("✓ Frame preprocessing works")
        
        # Test object detection
        objects = extractor.detect_objects_by_color(test_frame)
        print(f"✓ Object detection works - found {len(objects)} objects")
        
        # Test mock game state
        game_state = create_mock_game_state()
        assert isinstance(game_state, dict), "Game state should be a dictionary"
        assert "objects" in game_state, "Game state should contain objects"
        print("✓ Mock game state creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Object extractor test failed: {e}")
        traceback.print_exc()
        return False

def test_prompt_builder():
    """Test the prompt builder component"""
    print("\n=== Testing Prompt Builder ===")
    
    try:
        # Create prompt builder
        builder = PromptBuilder()
        print("✓ Prompt builder created successfully")
        
        # Create game context
        context = GameContext(
            episode=1,
            step=100,
            total_steps=1000,
            score=1500.0,
            time_remaining=300.0,
            difficulty="normal",
            player_status="active"
        )
        print("✓ Game context created")
        
        # Create mock objects
        objects = [
            GameObject("player", 32, 48, 16, 16, hp=5),
            GameObject("boss", 200, 100, 32, 32, hp=180, phase=1),
            GameObject("projectile", 150, 120, 8, 8, damage=1)
        ]
        print("✓ Mock objects created")
        
        # Build prompt
        prompt = builder.build_prompt(objects, context)
        assert isinstance(prompt, str), "Prompt should be a string"
        assert len(prompt) > 0, "Prompt should not be empty"
        assert "Hornet" in prompt, "Prompt should mention Hornet"
        print("✓ Prompt building works")
        
        # Test different scenarios
        scenarios = ["combat", "exploration", "platforming", "general"]
        for scenario in scenarios:
            scenario_prompt = builder.build_scenario_prompt(objects, context, scenario)
            assert isinstance(scenario_prompt, str), f"Scenario prompt for {scenario} should be a string"
            assert len(scenario_prompt) > 0, f"Scenario prompt for {scenario} should not be empty"
        print("✓ All scenario prompts work")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt builder test failed: {e}")
        traceback.print_exc()
        return False

def test_vlm_interface():
    """Test the VLM interface component"""
    print("\n=== Testing VLM Interface ===")
    
    try:
        # Test Mock VLM
        mock_vlm = MockVLM()
        print("✓ Mock VLM created successfully")
        
        # Create test image and prompt
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_prompt = "What action should Hornet take?"
        print("✓ Test image and prompt created")
        
        # Test action plan generation
        action_plan = mock_vlm.generate_action_plan(test_image, test_prompt)
        assert isinstance(action_plan, dict), "Action plan should be a dictionary"
        assert "subgoal" in action_plan, "Action plan should contain subgoal"
        assert "macro" in action_plan, "Action plan should contain macro action"
        print(f"✓ Mock VLM action plan generation works: {action_plan}")
        
        # Test multiple generations
        for i in range(3):
            plan = mock_vlm.generate_action_plan(test_image, test_prompt)
            assert isinstance(plan, dict), f"Action plan {i} should be a dictionary"
        print("✓ Multiple action plan generations work")
        
        return True
        
    except Exception as e:
        print(f"✗ VLM interface test failed: {e}")
        traceback.print_exc()
        return False

def test_action_translator():
    """Test the action translator component"""
    print("\n=== Testing Action Translator ===")
    
    try:
        # Create action translator
        translator = ActionTranslator()
        print("✓ Action translator created successfully")
        
        # Test parsing simple macro actions
        test_macros = [
            "MoveRight + Jump",
            "Dash + Attack",
            "JumpAttack + DownSlash",
            "MoveLeft + DashAttack"
        ]
        
        for macro in test_macros:
            steps = translator.parse_macro_action(macro)
            assert isinstance(steps, list), f"Parsed steps for '{macro}' should be a list"
            assert len(steps) > 0, f"Parsed steps for '{macro}' should not be empty"
            for step in steps:
                assert isinstance(step, ActionStep), f"Step should be ActionStep instance"
                assert step.action in translator.action_vocab, f"Action '{step.action}' should be in vocabulary"
        print("✓ Macro action parsing works")
        
        # Test action sequence generation
        for macro in test_macros:
            sequence = translator.generate_action_sequence(macro)
            assert isinstance(sequence, list), f"Action sequence for '{macro}' should be a list"
            assert len(sequence) > 0, f"Action sequence for '{macro}' should not be empty"
        print("✓ Action sequence generation works")
        
        # Test action embeddings
        test_actions = ["MoveRight", "Jump", "Attack"]
        for action in test_actions:
            embedding = translator.get_action_embedding(action)
            assert isinstance(embedding, np.ndarray), f"Embedding for '{action}' should be numpy array"
            assert len(embedding) > 0, f"Embedding for '{action}' should not be empty"
        print("✓ Action embedding generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Action translator test failed: {e}")
        traceback.print_exc()
        return False

def test_rl_policy():
    """Test the RL policy component"""
    print("\n=== Testing RL Policy ===")
    
    try:
        # Test device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        # Create goal-conditioned agent
        state_dim = 64  # Example state dimension
        goal_dim = 128  # Example goal embedding dimension
        action_dim = 8  # Example action dimension
        
        agent = GoalConditionedAgent(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            learning_rate=1e-3,
            buffer_size=1000,
            batch_size=32
        )
        print("✓ Goal-conditioned agent created successfully")
        
        # Test experience storage
        state = np.random.randn(state_dim)
        goal = np.random.randn(goal_dim)
        action = np.random.randint(action_dim)
        reward = np.random.random()
        next_state = np.random.randn(state_dim)
        next_goal = np.random.randn(goal_dim)
        done = False
        
        agent.store_experience(state, goal, action, reward, next_state, next_goal, done)
        print("✓ Experience storage works")
        
        # Test action selection
        selected_action = agent.get_action(state, goal)
        assert isinstance(selected_action, int), "Selected action should be an integer"
        assert 0 <= selected_action < action_dim, f"Action should be in range [0, {action_dim-1}]"
        print(f"✓ Action selection works: {selected_action}")
        
        # Test training with batch
        # Add more experiences to create a batch
        for _ in range(50):
            agent.store_experience(
                np.random.randn(state_dim),
                np.random.randn(goal_dim),
                np.random.randint(action_dim),
                np.random.random(),
                np.random.randn(state_dim),
                np.random.randn(goal_dim),
                np.random.random() > 0.9
            )
        
        # Test training step
        loss = agent.train_step()
        if loss is not None:
            assert isinstance(loss, float), "Training loss should be a float"
            assert loss >= 0, "Training loss should be non-negative"
            print(f"✓ Training step works: loss = {loss:.4f}")
        else:
            print("✓ Training step attempted (insufficient data for training)")
        
        return True
        
    except Exception as e:
        print(f"✗ RL policy test failed: {e}")
        traceback.print_exc()
        return False

def test_component_integration():
    """Test basic integration between components"""
    print("\n=== Testing Component Integration ===")
    
    try:
        # Create all components
        extractor = SimpleObjectExtractor()
        builder = PromptBuilder()
        vlm = MockVLM()
        translator = ActionTranslator()
        
        print("✓ All components created")
        
        # Create test data
        test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        context = GameContext(episode=1, step=100, total_steps=1000)
        
        # Test pipeline: frame -> objects -> prompt -> VLM -> actions
        objects = extractor.detect_objects_by_color(test_frame)
        prompt = builder.build_prompt(objects, context)
        action_plan = vlm.generate_action_plan(test_frame, prompt)
        action_steps = translator.parse_macro_action(action_plan["macro"])
        
        assert len(action_steps) > 0, "Should have at least one action step"
        print(f"✓ Integration pipeline works: {action_plan['subgoal']} -> {len(action_steps)} actions")
        
        return True
        
    except Exception as e:
        print(f"✗ Component integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all component tests"""
    print("=== Silksong VLA Model Component Tests ===\n")
    
    tests = [
        ("Object Extractor", test_object_extractor),
        ("Prompt Builder", test_prompt_builder),
        ("VLM Interface", test_vlm_interface),
        ("Action Translator", test_action_translator),
        ("RL Policy", test_rl_policy),
        ("Component Integration", test_component_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        results[test_name] = test_func()
    
    # Print summary
    print("\n=== Test Summary ===")
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Components are ready for integration.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
