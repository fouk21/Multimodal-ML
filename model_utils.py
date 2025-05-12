import clip
import torch
from PIL import Image

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def preprocess_image(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def encode_texts(model, device, text_prompts):
    return clip.tokenize(text_prompts).to(device)
