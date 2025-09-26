"""
Object Feature Extractor for Silksong RL Agent

This module handles extracting structured object information from game frames
using computer vision techniques. For now, we'll use a simplified approach
that can be upgraded to Cutie or other few-shot object extractors later.
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GameObject:
    """Represents a detected game object"""
    type: str  # "player", "boss", "projectile", "hazard", "platform", etc.
    x: float
    y: float
    width: float = 10.0
    height: float = 10.0
    
    # Dynamic properties
    vel_x: float = 0.0
    vel_y: float = 0.0
    
    # Game-specific properties
    hp: Optional[int] = None
    phase: Optional[int] = None
    damage: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "vel_x": self.vel_x, "vel_y": self.vel_y
        }
        
        if self.hp is not None:
            result["hp"] = self.hp
        if self.phase is not None:
            result["phase"] = self.phase
        if self.damage is not None:
            result["damage"] = self.damage
            
        return result

class SimpleObjectExtractor:
    """
    Simplified object extractor using color-based detection and contour analysis.
    This can be replaced with Cutie or other advanced models later.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (64, 64)):
        self.input_size = input_size
        
        # Color ranges for different objects (BGR format)
        # These are example values and should be calibrated for actual game
        self.color_ranges = {
            "player": {
                "lower": np.array([0, 100, 200]),    # Light blue/white for Hornet
                "upper": np.array([50, 150, 255])
            },
            "boss": {
                "lower": np.array([0, 0, 150]),      # Red/purple for bosses
                "upper": np.array([50, 50, 255])
            },
            "projectile": {
                "lower": np.array([0, 200, 200]),    # Cyan for projectiles
                "upper": np.array([50, 255, 255])
            },
            "hazard": {
                "lower": np.array([0, 0, 0]),        # Black/dark for hazards
                "upper": np.array([50, 50, 50])
            }
        }
        
        # Object tracking
        self.previous_objects = {}
        self.frame_count = 0
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for object detection"""
        # Resize to input size
        resized = cv2.resize(frame, self.input_size)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        return blurred
    
    def detect_objects_by_color(self, frame: np.ndarray) -> List[GameObject]:
        """Detect objects using color-based segmentation"""
        objects = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for obj_type, color_range in self.color_ranges.items():
            # Convert BGR to HSV
            lower_bgr = color_range["lower"]
            upper_bgr = color_range["upper"]
            
            # Simple approximation for HSV conversion
            # In practice, you'd want to convert these ranges properly
            mask = cv2.inRange(frame, lower_bgr, upper_bgr)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < 50:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Create game object
                obj = GameObject(
                    type=obj_type,
                    x=float(center_x),
                    y=float(center_y),
                    width=float(w),
                    height=float(h)
                )
                
                # Add game-specific properties based on object type
                if obj_type == "player":
                    obj.hp = 5  # Assume full health
                elif obj_type == "boss":
                    obj.hp = 180
                    obj.phase = 1
                elif obj_type == "projectile":
                    obj.damage = 1
                
                objects.append(obj)
        
        return objects
    
    def calculate_velocities(self, current_objects: List[GameObject]) -> List[GameObject]:
        """Calculate velocities based on previous positions"""
        if self.frame_count == 0:
            return current_objects
        
        # Match objects with previous frame
        for current_obj in current_objects:
            # Find closest object of same type in previous frame
            min_distance = float('inf')
            matched_prev_obj = None
            
            for prev_obj in self.previous_objects.get(current_obj.type, []):
                distance = np.sqrt(
                    (current_obj.x - prev_obj.x)**2 + 
                    (current_obj.y - prev_obj.y)**2
                )
                if distance < min_distance and distance < 50:  # Max matching distance
                    min_distance = distance
                    matched_prev_obj = prev_obj
            
            if matched_prev_obj:
                # Calculate velocity
                dt = 1.0  # Assume 1 frame time difference
                current_obj.vel_x = (current_obj.x - matched_prev_obj.x) / dt
                current_obj.vel_y = (current_obj.y - matched_prev_obj.y) / dt
        
        return current_objects
    
    def extract_objects(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract structured object information from frame"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Detect objects
        objects = self.detect_objects_by_color(processed_frame)
        
        # Calculate velocities
        objects = self.calculate_velocities(objects)
        
        # Organize by type
        object_dict = {}
        for obj in objects:
            if obj.type not in object_dict:
                object_dict[obj.type] = []
            object_dict[obj.type].append(obj.to_dict())
        
        # Store for next frame velocity calculation
        self.previous_objects = {}
        for obj in objects:
            if obj.type not in self.previous_objects:
                self.previous_objects[obj.type] = []
            self.previous_objects[obj.type].append(obj)
        
        self.frame_count += 1
        
        return {
            "objects": object_dict,
            "frame_count": self.frame_count,
            "input_size": self.input_size
        }

class CNNObjectExtractor(nn.Module):
    """
    CNN-based object extractor that can be trained to detect objects.
    This is a placeholder for more advanced models like Cutie.
    """
    
    def __init__(self, input_channels: int = 3, num_object_types: int = 4):
        super().__init__()
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Object detection head
        self.detection_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_object_types * 6)  # x, y, w, h, vel_x, vel_y for each type
        )
        
        self.num_object_types = num_object_types
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        batch_size = features.shape[0]
        features_flat = features.view(batch_size, -1)
        
        detections = self.detection_head(features_flat)
        detections = detections.view(batch_size, self.num_object_types, 6)
        
        return detections

# Utility functions
def create_mock_game_state() -> Dict[str, Any]:
    """Create a mock game state for testing"""
    return {
        "player": {
            "x": 100, "y": 95, "hp": 4, "vel_x": 0, "vel_y": 0
        },
        "boss": {
            "x": 130, "y": 95, "hp": 180, "phase": 1, "vel_x": 0, "vel_y": 0
        },
        "projectiles": [
            {"x": 110, "y": 90, "dx": 2, "dy": 0, "damage": 1}
        ],
        "hazards": [],
        "platforms": []
    }

def test_object_extractor():
    """Test the object extractor"""
    # Create a simple test frame
    test_frame = np.zeros((144, 256, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    # Player (light blue)
    cv2.rectangle(test_frame, (90, 85), (110, 105), (200, 150, 100), -1)
    
    # Boss (red)
    cv2.rectangle(test_frame, (120, 85), (140, 105), (255, 50, 50), -1)
    
    # Projectile (cyan)
    cv2.circle(test_frame, (110, 90), 5, (255, 255, 200), -1)
    
    # Test extraction
    extractor = SimpleObjectExtractor(input_size=(64, 64))
    result = extractor.extract_objects(test_frame)
    
    print("Object Extraction Result:")
    print(result)
    
    return result

if __name__ == "__main__":
    test_object_extractor()
