import pyautogui
import keyconfig
import time
import threading

# State management for tracking key presses and jumps
class ControlState:
    def __init__(self):
        self.keys_pressed = set()
        self.is_jumping = False
        self.jump_start_time = 0
        self.can_double_jump = True
        self.has_double_jumped = False
        self.movement_threads = {}
        
control_state = ControlState()

# Existing functions
def UseUpTool():
    pyautogui.keyDown(keyconfig.UP)
    pyautogui.press(keyconfig.TOOL)
    pyautogui.keyUp(keyconfig.UP)

def UseMidTool():
    pyautogui.press(keyconfig.TOOL)

def UseDownTool():
    pyautogui.keyDown(keyconfig.DOWN)
    pyautogui.press(keyconfig.TOOL)
    pyautogui.keyUp(keyconfig.DOWN)

def Dash():
    pyautogui.press(keyconfig.DASH)

def Hook():
    pyautogui.press(keyconfig.HOOK)

def Attack():
    pyautogui.press(keyconfig.ATTACK)
    
def DownSlash():
    pyautogui.keyDown(keyconfig.DOWN)
    pyautogui.press(keyconfig.ATTACK)
    pyautogui.keyUp(keyconfig.DOWN)

# Basic movement controls with press and release functionality
def MoveLeft():
    """Start moving left"""
    if keyconfig.LEFT not in control_state.keys_pressed:
        pyautogui.keyDown(keyconfig.LEFT)
        control_state.keys_pressed.add(keyconfig.LEFT)

def MoveRight():
    """Start moving right"""
    if keyconfig.RIGHT not in control_state.keys_pressed:
        pyautogui.keyDown(keyconfig.RIGHT)
        control_state.keys_pressed.add(keyconfig.RIGHT)

def MoveUp():
    """Start moving up"""
    if keyconfig.UP not in control_state.keys_pressed:
        pyautogui.keyDown(keyconfig.UP)
        control_state.keys_pressed.add(keyconfig.UP)

def MoveDown():
    """Start moving down"""
    if keyconfig.DOWN not in control_state.keys_pressed:
        pyautogui.keyDown(keyconfig.DOWN)
        control_state.keys_pressed.add(keyconfig.DOWN)

def StopMoveLeft():
    """Stop moving left"""
    if keyconfig.LEFT in control_state.keys_pressed:
        pyautogui.keyUp(keyconfig.LEFT)
        control_state.keys_pressed.remove(keyconfig.LEFT)

def StopMoveRight():
    """Stop moving right"""
    if keyconfig.RIGHT in control_state.keys_pressed:
        pyautogui.keyUp(keyconfig.RIGHT)
        control_state.keys_pressed.remove(keyconfig.RIGHT)

def StopMoveUp():
    """Stop moving up"""
    if keyconfig.UP in control_state.keys_pressed:
        pyautogui.keyUp(keyconfig.UP)
        control_state.keys_pressed.remove(keyconfig.UP)

def StopMoveDown():
    """Stop moving down"""
    if keyconfig.DOWN in control_state.keys_pressed:
        pyautogui.keyUp(keyconfig.DOWN)
        control_state.keys_pressed.remove(keyconfig.DOWN)

# Jump control with variable height (hold to jump higher)
def StartJump():
    """Start a jump with variable height control"""
    if not control_state.is_jumping:
        pyautogui.keyDown(keyconfig.JUMP)
        control_state.is_jumping = True
        control_state.jump_start_time = time.time()
        control_state.can_double_jump = True
        control_state.has_double_jumped = False

def StopJump():
    """Stop jump - releases jump key for variable height control"""
    if control_state.is_jumping:
        pyautogui.keyUp(keyconfig.JUMP)
        control_state.is_jumping = False
        # Allow double jump after a short delay
        threading.Timer(0.1, lambda: setattr(control_state, 'can_double_jump', True)).start()

# Double jump functionality
def DoubleJump():
    """Perform a double jump if available"""
    if control_state.can_double_jump and not control_state.has_double_jumped:
        pyautogui.press(keyconfig.JUMP)
        control_state.has_double_jumped = True
        control_state.can_double_jump = False

# Continuous movement controls for holding keys down
def HoldMovement(direction, duration):
    """Hold a movement key for a specified duration"""
    def hold_key():
        if direction == 'left':
            MoveLeft()
        elif direction == 'right':
            MoveRight()
        elif direction == 'up':
            MoveUp()
        elif direction == 'down':
            MoveDown()
        
        time.sleep(duration)
        
        if direction == 'left':
            StopMoveLeft()
        elif direction == 'right':
            StopMoveRight()
        elif direction == 'up':
            StopMoveUp()
        elif direction == 'down':
            StopMoveDown()
    
    thread = threading.Thread(target=hold_key)
    thread.daemon = True
    thread.start()
    return thread

# Jump with variable height control
def VariableJump(duration):
    """Jump for a variable duration to control jump height"""
    def variable_height_jump():
        StartJump()
        time.sleep(duration)
        StopJump()
    
    thread = threading.Thread(target=variable_height_jump)
    thread.daemon = True
    thread.start()
    return thread

# Utility functions
def StopAllMovement():
    """Stop all movement keys"""
    for key in [keyconfig.LEFT, keyconfig.RIGHT, keyconfig.UP, keyconfig.DOWN]:
        if key in control_state.keys_pressed:
            pyautogui.keyUp(key)
    control_state.keys_pressed.clear()

def ResetJumpState():
    """Reset jump state (useful when landing on ground)"""
    if control_state.is_jumping:
        pyautogui.keyUp(keyconfig.JUMP)
    control_state.is_jumping = False
    control_state.can_double_jump = True
    control_state.has_double_jumped = False

def GetControlState():
    """Get current control state for debugging/monitoring"""
    return {
        'keys_pressed': list(control_state.keys_pressed),
        'is_jumping': control_state.is_jumping,
        'can_double_jump': control_state.can_double_jump,
        'has_double_jumped': control_state.has_double_jumped
    }

# Advanced movement combinations
def JumpLeft(duration=None):
    """Jump while moving left"""
    MoveLeft()
    if duration:
        VariableJump(duration)
    else:
        StartJump()
    return threading.Timer(0.1, lambda: None)

def JumpRight(duration=None):
    """Jump while moving right"""
    MoveRight()
    if duration:
        VariableJump(duration)
    else:
        StartJump()
    return threading.Timer(0.1, lambda: None)

def DashAttack():
    """Dash followed by immediate attack"""
    Dash()
    time.sleep(0.1)
    Attack()

def JumpAttack():
    """Jump followed by air attack"""
    StartJump()
    time.sleep(0.2)
    Attack()
    StopJump()

# Silksong-specific controls (can be expanded based on game mechanics)
def Bind():
    """Focus/heal ability (if mapped to a key)"""
    # This would need to be added to keyconfig.py if needed
    # pyautogui.press(keyconfig.FOCUS)
    pyautogui.keyDown(keyconfig.BIND)
    time.sleep(0.1)
    pyautogui.keyUp(keyconfig.BIND)

def QuickMap(duration=1):
    """Quick map access"""  
    # This would need to be added to keyconfig.py if needed
    # pyautogui.press(keyconfig.MAP)
    pyautogui.keyDown(keyconfig.QUICK_MAP)
    time.sleep(duration)
    pyautogui.keyUp(keyconfig.QUICK_MAP)

# Cleanup function
def Cleanup():
    """Release all keys and reset state"""
    StopAllMovement()
    ResetJumpState()
    
    # Ensure all keys are released
    for key in [keyconfig.LEFT, keyconfig.RIGHT, keyconfig.UP, keyconfig.DOWN, 
               keyconfig.JUMP, keyconfig.ATTACK, keyconfig.DASH, keyconfig.HOOK, 
               keyconfig.TOOL]:
        try:
            pyautogui.keyUp(key)
        except:
            pass