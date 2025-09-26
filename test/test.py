from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

# pick a small VLA model (replace with exact Hugging Face id)
model_id = "HuggingFaceTB/SmolLM3-3B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# load your Silksong screenshot
image = Image.open("test.jpg").convert("RGB")

prompt = """
You are controlling Hornet in Silksong.
Your goal is to beat the boss in the current area.
Make sure to take minimum damage.
And avoid making the bossfight too long.
Look at the screenshot and decide the next action.
Output only one line in this format:

Action: <macro action>
"""

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
print(decoded)

