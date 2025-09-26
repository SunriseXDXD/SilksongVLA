import psutil
import time
import win32gui
from typing import Optional, Tuple
import logging
import os

# Import configuration
try:
    from config.process_detection_config import ProcessDetectionConfig
except ImportError:
    # Fallback configuration if config file not found
    class ProcessDetectionConfig:
        def __init__(self):
            self.silksong_process_names = ['Silksong.exe', 'HollowKnightSilksong.exe']
            self.silksong_window_titles = ['Hollow Knight: Silksong', 'Silksong']
            self.check_interval = 1.0
            self.max_wait_time = 30.0
            self.require_focus = True
            self.auto_restart_training = True
            self.enable_logging = True
            self.log_level = 'INFO'
            self.platform = 'windows'
            self.check_window_class = True
            self.silksong_window_classes = ['UnityWndClass', 'UnrealWindow']

# Configure logging
config = ProcessDetectionConfig()
if config.enable_logging:
    logging.basicConfig(level=getattr(logging, config.log_level.upper()))
logger = logging.getLogger(__name__)

class ProcessDetector:
    def __init__(self, config_file=None):
        # Load configuration
        if config_file and os.path.exists(config_file):
            self.config = ProcessDetectionConfig()
            self.config.load_config(config_file)
        else:
            self.config = ProcessDetectionConfig()
        
        # Use configuration values
        self.silksong_process_names = self.config.silksong_process_names
        self.silksong_window_titles = self.config.silksong_window_titles
        self.check_interval = self.config.check_interval
        self.max_wait_time = self.config.max_wait_time
        self.require_focus = self.config.require_focus
        self.auto_restart_training = self.config.auto_restart_training
        
        # State tracking
        self.last_detection_time = 0
        self.is_running = False
        self.window_handle = None
        
    def is_silksong_running(self) -> bool:
        """Check if Silksong process is currently running"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_info = proc.info
                    if proc_info['name'] in self.silksong_process_names:
                        logger.info(f"Found Silksong process: {proc_info['name']} (PID: {proc_info['pid']})")
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.error(f"Error checking processes: {e}")
        
        return False
    
    def find_silksong_window(self) -> Optional[int]:
        """Find Silksong window handle"""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                for title in self.silksong_window_titles:
                    if title.lower() in window_title.lower():
                        windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            # Return the first matching window
            return windows[0]
        return None
    
    def is_window_focused(self, window_handle: int) -> bool:
        """Check if the specified window has focus"""
        try:
            foreground_window = win32gui.GetForegroundWindow()
            return foreground_window == window_handle
        except Exception as e:
            logger.error(f"Error checking window focus: {e}")
            return False
    
    def get_window_rect(self, window_handle: int) -> Optional[Tuple[int, int, int, int]]:
        """Get window rectangle (left, top, right, bottom)"""
        try:
            return win32gui.GetWindowRect(window_handle)
        except Exception as e:
            logger.error(f"Error getting window rect: {e}")
            return None
    
    def is_game_ready(self) -> Tuple[bool, Optional[int]]:
        """Check if game is running and optionally focused"""
        # Check if process is running
        if not self.is_silksong_running():
            logger.info("Silksong process not found")
            return False, None
        
        # Find game window
        window_handle = self.find_silksong_window()
        if not window_handle:
            logger.info("Silksong window not found")
            return False, None
        
        # Check focus if required
        if self.require_focus and not self.is_window_focused(window_handle):
            logger.info("Silksong window is not focused")
            return False, window_handle
        
        logger.info("Silksong is ready")
        return True, window_handle
    
    def wait_for_game(self, timeout: Optional[float] = None) -> Tuple[bool, Optional[int]]:
        """Wait for Silksong to start and be ready"""
        if timeout is None:
            timeout = self.max_wait_time
        
        start_time = time.time()
        
        logger.info(f"Waiting for Silksong to start (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            is_ready, window_handle = self.is_game_ready()
            
            if is_ready:
                self.is_running = True
                self.window_handle = window_handle
                self.last_detection_time = time.time()
                return True, window_handle
            
            time.sleep(self.check_interval)
        
        logger.warning(f"Silksong did not start within {timeout} seconds")
        return False, None
    
    def monitor_game_status(self, callback=None):
        """Continuously monitor game status"""
        logger.info("Starting game status monitoring...")
        
        while True:
            was_running = self.is_running
            
            is_ready, window_handle = self.is_game_ready()
            
            if is_ready:
                if not was_running:
                    logger.info("Silksong started")
                    self.is_running = True
                    self.window_handle = window_handle
                    if callback:
                        callback('started', window_handle)
            else:
                if was_running:
                    logger.info("Silksong stopped")
                    self.is_running = False
                    self.window_handle = None
                    if callback:
                        callback('stopped', None)
            
            self.last_detection_time = time.time()
            time.sleep(self.check_interval)
    
    def get_game_info(self) -> dict:
        """Get comprehensive game information"""
        info = {
            'is_running': self.is_running,
            'window_handle': self.window_handle,
            'last_detection': self.last_detection_time,
            'process_found': self.is_silksong_running(),
            'window_found': self.find_silksong_window() is not None,
            'window_focused': False,
            'window_rect': None
        }
        
        if self.window_handle:
            info['window_focused'] = self.is_window_focused(self.window_handle)
            info['window_rect'] = self.get_window_rect(self.window_handle)
        
        return info
    
    def set_focus_requirement(self, require_focus: bool):
        """Set whether game window needs to be focused"""
        self.require_focus = require_focus
        logger.info(f"Focus requirement set to: {require_focus}")
    
    def add_process_name(self, process_name: str):
        """Add additional process name to check"""
        if process_name not in self.silksong_process_names:
            self.silksong_process_names.append(process_name)
            logger.info(f"Added process name: {process_name}")
    
    def add_window_title(self, window_title: str):
        """Add additional window title to check"""
        if window_title not in self.silksong_window_titles:
            self.silksong_window_titles.append(window_title)
            logger.info(f"Added window title: {window_title}")

# Convenience functions
def is_silksong_running() -> bool:
    """Quick check if Silksong is running"""
    detector = ProcessDetector()
    return detector.is_silksong_running()

def wait_for_silksong(timeout: float = 30.0) -> bool:
    """Wait for Silksong to start"""
    detector = ProcessDetector()
    success, _ = detector.wait_for_game(timeout)
    return success

def get_silksong_window_info() -> dict:
    """Get Silksong window information"""
    detector = ProcessDetector()
    return detector.get_game_info()

# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = ProcessDetector()
    
    print("Checking if Silksong is running...")
    is_running = detector.is_silksong_running()
    print(f"Silksong running: {is_running}")
    
    if is_running:
        window_handle = detector.find_silksong_window()
        if window_handle:
            rect = detector.get_window_rect(window_handle)
            focused = detector.is_window_focused(window_handle)
            print(f"Window handle: {window_handle}")
            print(f"Window rect: {rect}")
            print(f"Window focused: {focused}")
    
    print("\nGame info:")
    info = detector.get_game_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nWaiting for Silksong (10 second timeout)...")
    success, handle = detector.wait_for_game(10.0)
    print(f"Wait result: {success}, handle: {handle}")
