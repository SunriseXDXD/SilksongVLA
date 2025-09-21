# Screen capture configuration
class ScreenConfig:
    def __init__(self):
        self.screen_width = 2560  # Updated for 2560x1440 screen
        self.screen_height = 1440  # Updated for 2560x1440 screen
        
        # Fullscreen Silksong capture settings
        self.capture_width = 2560  # Full screen width
        self.capture_height = 1440  # Full screen height
        self.capture_x = 0  # Start from left edge (fullscreen)
        self.capture_y = 0  # Start from top edge (fullscreen)
        
        # Capture settings
        self.fps = 60
        self.frame_delay = 1.0 / self.fps
        
        # Preprocessing parameters
        self.target_width = 84  # RL input size
        self.target_height = 84
        self.grayscale = True
        self.normalize = True
        
        # Optional: Center crop settings if needed
        self.center_crop = False
        self.crop_width = 1920  # If center cropping, use this width
        self.crop_height = 1080  # If center cropping, use this height