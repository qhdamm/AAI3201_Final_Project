from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("input.jpg")
inputs = processor(images=image, return_tensors="pt", padding=True)

outputs = model.get_text_features(**inputs)
print(outputs)
