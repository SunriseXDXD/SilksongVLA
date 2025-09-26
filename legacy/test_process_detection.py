#!/usr/bin/env python3
"""
Test script for Silksong process detection
"""

import time
import sys
from process_detector import ProcessDetector

def test_process_detection():
    """Test the process detection functionality"""
    print("=== Silksong Process Detection Test ===\n")
    
    # Create detector
    detector = ProcessDetector()
    
    print("1. Testing basic process detection...")
    print(f"   Checking for process names: {detector.silksong_process_names}")
    print(f"   Checking for window titles: {detector.silksong_window_titles}")
    
    # Check if Silksong is running
    is_running = detector.is_silksong_running()
    print(f"   Silksong process running: {is_running}")
    
    if is_running:
        print("   ✓ Silksong process found!")
    else:
        print("   ✗ Silksong process not found")
    
    print("\n2. Testing window detection...")
    window_handle = detector.find_silksong_window()
    print(f"   Silksong window handle: {window_handle}")
    
    if window_handle:
        print("   ✓ Silksong window found!")
        
        # Get window info
        rect = detector.get_window_rect(window_handle)
        focused = detector.is_window_focused(window_handle)
        print(f"   Window rectangle: {rect}")
        print(f"   Window focused: {focused}")
    else:
        print("   ✗ Silksong window not found")
    
    print("\n3. Testing game readiness check...")
    is_ready, window_handle = detector.is_game_ready()
    print(f"   Game ready: {is_ready}")
    print(f"   Window handle: {window_handle}")
    
    if is_ready:
        print("   ✓ Game is ready for training!")
    else:
        print("   ✗ Game is not ready")
    
    print("\n4. Getting comprehensive game info...")
    game_info = detector.get_game_info()
    print("   Game information:")
    for key, value in game_info.items():
        print(f"     {key}: {value}")
    
    print("\n5. Testing configuration...")
    config_summary = detector.config.get_config_summary()
    print("   Configuration summary:")
    for key, value in config_summary.items():
        if isinstance(value, list):
            print(f"     {key}: {', '.join(value[:3])}{'...' if len(value) > 3 else ''}")
        else:
            print(f"     {key}: {value}")
    
    print("\n=== Test Complete ===")
    
    return is_ready

def test_wait_for_game():
    """Test waiting for game to start"""
    print("\n=== Testing Wait for Game ===")
    print("This will wait up to 10 seconds for Silksong to start...")
    print("Start Silksong now to test this functionality!")
    
    detector = ProcessDetector()
    success, handle = detector.wait_for_game(10.0)
    
    if success:
        print(f"✓ Game started successfully! Window handle: {handle}")
    else:
        print("✗ Game did not start within timeout")
    
    return success

def main():
    """Main test function"""
    print("Silksong Process Detection Test Suite")
    print("====================================")
    
    try:
        # Test basic detection
        is_ready = test_process_detection()
        
        if not is_ready:
            # Ask user if they want to test waiting for game
            response = input("\nWould you like to test waiting for game to start? (y/n): ")
            if response.lower() in ['y', 'yes']:
                test_wait_for_game()
        
        print("\n=== All Tests Completed ===")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
