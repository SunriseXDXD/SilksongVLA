"""
Vision-Language Model Interface for Silksong RL Agent

This module provides an interface to various Vision-Language Models
that can process game frames and generate action plans.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import transformers
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMInterface:
    """
    Interface for Vision-Language Models that can process game frames
    and generate action plans.
    """
    
    def __init__(self, 
                 model_name: str = "blip2-opt-2.7b",
                 device: str = "auto",
                 torch_dtype: Optional[torch.dtype] = None):
        """
        Initialize VLM interface
        
        Args:
            model_name: Name of the VLM model to use
            device: Device to run the model on
            torch_dtype: Data type for model tensors
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.torch_dtype = torch_dtype or torch.float16
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Model-specific settings
        self.model_configs = {
            "blip2-opt-2.7b": {
                "model_class": AutoModelForVision2Seq,
                "processor_class": AutoProcessor,
                "supports_images": True,
                "max_length": 512
            },
            "llava-1.5-7b": {
                "model_class": AutoModelForVision2Seq,
                "processor_class": AutoProcessor,
                "supports_images": True,
                "max_length": 512
            },
            "git-large-coco": {
                "model_class": AutoModelForVision2Seq,
                "processor_class": AutoProcessor,
                "supports_images": True,
                "max_length": 512
            }
        }
        
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load the VLM model and processor"""
        try:
            config = self.model_configs.get(self.model_name, self.model_configs["blip2-opt-2.7b"])
            
            logger.info(f"Loading VLM model: {self.model_name}")
            
            # Load processor
            self.processor = config["processor_class"].from_pretrained(self.model_name)
            
            # Load model
            self.model = config["model_class"].from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != "cpu" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"VLM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Image.Image:
        """Preprocess image for VLM input"""
        if isinstance(image, Image.Image):
            return image
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # BGR to RGB
                image = image[:, :, ::-1]
            elif image.shape[2] == 4:  # BGRA to RGB
                image = image[:, :, [2, 1, 0]]
        
        return Image.fromarray(image)
    
    def generate_action_plan(self, 
                           image: np.ndarray,
                           prompt: str,
                           max_new_tokens: int = 100,
                           temperature: float = 0.7,
                           do_sample: bool = True) -> Dict[str, str]:
        """
        Generate action plan from image and prompt
        
        Args:
            image: Game frame as numpy array
            prompt: Text prompt describing the task
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary containing subgoal and macro action
        """
        try:
            # Preprocess image
            pil_image = self.preprocess_image(image)
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract the generated part (remove the prompt)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Parse response to extract subgoal and macro
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error generating action plan: {e}")
            return {"subgoal": "error", "macro": "no_action"}
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse VLM response to extract subgoal and macro"""
        lines = response.strip().split('\n')
        result = {"subgoal": "", "macro": ""}
        
        for line in lines:
            line = line.strip().lower()
            if line.startswith("subgoal:"):
                result["subgoal"] = line.replace("subgoal:", "").strip()
            elif line.startswith("macro:"):
                result["macro"] = line.replace("macro:", "").strip()
        
        # Fallback parsing if structured format not found
        if not result["subgoal"] and not result["macro"]:
            # Try to extract action keywords
            action_keywords = ["move", "jump", "dash", "attack", "slash", "avoid", "dodge"]
            for keyword in action_keywords:
                if keyword in response.lower():
                    result["subgoal"] = keyword
                    result["macro"] = keyword.capitalize()
                    break
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "supports_images": self.model_configs.get(self.model_name, {}).get("supports_images", False),
            "max_length": self.model_configs.get(self.model_name, {}).get("max_length", 512)
        }

class MockVLM(VLMInterface):
    """
    Mock VLM for testing purposes when real models are not available
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "mock_vlm"
        
        # Predefined responses for testing
        self.responses = [
            {"subgoal": "avoid projectile", "macro": "JumpRight + Dash"},
            {"subgoal": "attack boss", "macro": "MoveRight + Attack"},
            {"subgoal": "dodge attack", "macro": "Dash + JumpAttack"},
            {"subgoal": "position advantage", "macro": "MoveLeft + Jump"},
            {"subgoal": "counter attack", "macro": "DownSlash + DashAttack"}
        ]
        self.response_index = 0
    
    def generate_action_plan(self, 
                           image: np.ndarray,
                           prompt: str,
                           max_new_tokens: int = 100,
                           temperature: float = 0.7,
                           do_sample: bool = True) -> Dict[str, str]:
        """Generate mock action plan"""
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        
        logger.info(f"Mock VLM response: {response}")
        return response

class LocalVLM(VLMInterface):
    """
    Local VLM implementation using smaller models that can run locally
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Use a smaller model that can run locally
        model_name = model_path or "Salesforce/blip2-opt-2.7b"
        super().__init__(model_name=model_name, device="cpu")

def test_vlm_interface():
    """Test the VLM interface"""
    print("Testing VLM Interface...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test prompt
    test_prompt = """You are controlling Hornet in Silksong. 
Game state:
- Player: at (100, 95) HP: 4
- Boss: at (130, 95) HP: 180, Phase: 1
- Projectile: at (110, 90) moving (2, 0)

Decide the next action plan.
Output:
Subgoal: <short description of immediate objective>
Macro: <sequence of actions>"""
    
    try:
        # Try to use mock VLM first for testing
        vlm = MockVLM()
        print(f"Using mock VLM: {vlm.model_name}")
        
        # Generate action plan
        result = vlm.generate_action_plan(test_image, test_prompt)
        print(f"Generated action plan: {result}")
        
        # Get model info
        info = vlm.get_model_info()
        print(f"Model info: {info}")
        
    except Exception as e:
        print(f"Error testing VLM interface: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vlm_interface()
