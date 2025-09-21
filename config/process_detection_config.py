"""
Process Detection Configuration for Silksong RL Training

This configuration file contains settings for detecting the Silksong game process
and window. These settings can be customized based on your specific setup.
"""

class ProcessDetectionConfig:
    def __init__(self):
        # Process names to check for Silksong
        # Add any process names that Silksong might use on your system
        self.silksong_process_names = [
            'Silksong.exe',
            'HollowKnightSilksong.exe', 
            'TeamCherry.Silksong.exe',
            'silksong.exe',
            'hollowknight_silksong.exe',
            'Hollow Knight Silksong.exe',
            'Silksong-Win64-Shipping.exe'  # Unreal Engine builds
        ]
        
        # Window titles to check for Silksong
        # Add any window titles that Silksong might use on your system
        self.silksong_window_titles = [
            'Hollow Knight: Silksong',
            'Silksong',
            'Team Cherry - Silksong',
            'Hollow Knight: Silksong*',
            'Silksong - Unity',
            'Silksong (Development)',
            'UnityPlayerWindow'  # Unity builds
        ]
        
        # Detection behavior settings
        self.check_interval = 1.0  # seconds between process checks
        self.max_wait_time = 60.0  # maximum time to wait for game to start (seconds)
        self.require_focus = True  # whether game window needs to be in focus
        self.auto_restart_training = True  # automatically resume when game comes back
        
        # Fallback behavior settings
        self.fallback_wait_interval = 2.0  # seconds between fallback checks
        self.stop_all_actions_on_disconnect = True  # stop character movement when game disconnects
        
        # Logging settings
        self.enable_logging = True
        self.log_level = 'INFO'  # DEBUG, INFO, WARNING, ERROR
        
        # Platform-specific settings
        self.platform = 'windows'  # windows, linux, mac
        
        # Windows-specific settings
        if self.platform == 'windows':
            self.check_window_class = True  # check window class names
            self.silksong_window_classes = [
                'UnityWndClass',  # Unity games
                'UnrealWindow',  # Unreal Engine games
                'SDL_app'  # SDL-based games
            ]
        
        # Linux-specific settings (if needed)
        elif self.platform == 'linux':
            self.use_xlib = True
            self.check_wm_class = True
        
        # Mac-specific settings (if needed)
        elif self.platform == 'mac':
            self.use_pyobjc = True
            self.check_bundle_id = True
            self.silksong_bundle_ids = [
                'com.teamcherry.silksong',
                'com.hollowknight.silksong'
            ]
    
    def add_process_name(self, process_name):
        """Add a custom process name to check"""
        if process_name not in self.silksong_process_names:
            self.silksong_process_names.append(process_name)
            print(f"Added process name: {process_name}")
    
    def add_window_title(self, window_title):
        """Add a custom window title to check"""
        if window_title not in self.silksong_window_titles:
            self.silksong_window_titles.append(window_title)
            print(f"Added window title: {window_title}")
    
    def remove_process_name(self, process_name):
        """Remove a process name from the check list"""
        if process_name in self.silksong_process_names:
            self.silksong_process_names.remove(process_name)
            print(f"Removed process name: {process_name}")
    
    def remove_window_title(self, window_title):
        """Remove a window title from the check list"""
        if window_title in self.silksong_window_titles:
            self.silksong_window_titles.remove(window_title)
            print(f"Removed window title: {window_title}")
    
    def get_config_summary(self):
        """Get a summary of current configuration"""
        return {
            'process_names': self.silksong_process_names,
            'window_titles': self.silksong_window_titles,
            'check_interval': self.check_interval,
            'max_wait_time': self.max_wait_time,
            'require_focus': self.require_focus,
            'auto_restart_training': self.auto_restart_training,
            'platform': self.platform
        }
    
    def save_config(self, filepath):
        """Save configuration to a file"""
        import json
        
        config_data = {
            'silksong_process_names': self.silksong_process_names,
            'silksong_window_titles': self.silksong_window_titles,
            'check_interval': self.check_interval,
            'max_wait_time': self.max_wait_time,
            'require_focus': self.require_focus,
            'auto_restart_training': self.auto_restart_training,
            'fallback_wait_interval': self.fallback_wait_interval,
            'stop_all_actions_on_disconnect': self.stop_all_actions_on_disconnect,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'platform': self.platform
        }
        
        if self.platform == 'windows':
            config_data['check_window_class'] = self.check_window_class
            config_data['silksong_window_classes'] = self.silksong_window_classes
        elif self.platform == 'linux':
            config_data['use_xlib'] = self.use_xlib
            config_data['check_wm_class'] = self.check_wm_class
        elif self.platform == 'mac':
            config_data['use_pyobjc'] = self.use_pyobjc
            config_data['check_bundle_id'] = self.check_bundle_id
            config_data['silksong_bundle_ids'] = self.silksong_bundle_ids
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath):
        """Load configuration from a file"""
        import json
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            self.silksong_process_names = config_data.get('silksong_process_names', self.silksong_process_names)
            self.silksong_window_titles = config_data.get('silksong_window_titles', self.silksong_window_titles)
            self.check_interval = config_data.get('check_interval', self.check_interval)
            self.max_wait_time = config_data.get('max_wait_time', self.max_wait_time)
            self.require_focus = config_data.get('require_focus', self.require_focus)
            self.auto_restart_training = config_data.get('auto_restart_training', self.auto_restart_training)
            self.fallback_wait_interval = config_data.get('fallback_wait_interval', self.fallback_wait_interval)
            self.stop_all_actions_on_disconnect = config_data.get('stop_all_actions_on_disconnect', self.stop_all_actions_on_disconnect)
            self.enable_logging = config_data.get('enable_logging', self.enable_logging)
            self.log_level = config_data.get('log_level', self.log_level)
            self.platform = config_data.get('platform', self.platform)
            
            if self.platform == 'windows':
                self.check_window_class = config_data.get('check_window_class', self.check_window_class)
                self.silksong_window_classes = config_data.get('silksong_window_classes', self.silksong_window_classes)
            elif self.platform == 'linux':
                self.use_xlib = config_data.get('use_xlib', self.use_xlib)
                self.check_wm_class = config_data.get('check_wm_class', self.check_wm_class)
            elif self.platform == 'mac':
                self.use_pyobjc = config_data.get('use_pyobjc', self.use_pyobjc)
                self.check_bundle_id = config_data.get('check_bundle_id', self.check_bundle_id)
                self.silksong_bundle_ids = config_data.get('silksong_bundle_ids', self.silksong_bundle_ids)
            
            print(f"Configuration loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Configuration file not found: {filepath}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = ProcessDetectionConfig()
    
    # Print current configuration
    print("Current Process Detection Configuration:")
    summary = config.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test adding custom process name
    print("\nAdding custom process name...")
    config.add_process_name("CustomSilksong.exe")
    
    # Save configuration
    print("\nSaving configuration...")
    config.save_config("process_detection_config.json")
    
    # Load configuration
    print("\nLoading configuration...")
    new_config = ProcessDetectionConfig()
    new_config.load_config("process_detection_config.json")
    
    print("\nLoaded configuration:")
    summary = new_config.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
