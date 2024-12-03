import os
from flask import Flask, request, jsonify
import requests
from PIL import Image, ImageEnhance, ImageChops
import torch
from diffusers import StableDiffusionPipeline
import io
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# Define the API URL and headers for Hugging Face model
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": "Bearer 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"}

# Function to query the Hugging Face model and get the image
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Ensure GPU is available for Stable Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe.to(device)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to compute similarity between image and text prompt using CLIP
def compute_clip_score(image, text):
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities
    return probs[0][0].item()  # Return the similarity score for the given text-image pair

# Function to enhance the image from the Stable Diffusion pipeline
def enhance_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    image_sharpened = enhancer.enhance(1.7)

    contrast_enhancer = ImageEnhance.Contrast(image_sharpened)
    image_contrast = contrast_enhancer.enhance(1.3)

    brightness_enhancer = ImageEnhance.Brightness(image_contrast)
    image_brightness = brightness_enhancer.enhance(1.1)

    return image_brightness

@app.route('/generate_image', methods=['POST'])
def generate_image():
    user_prompt = request.json.get('prompt')
    
    # Request image from Hugging Face FLUX model
    flux_image_bytes = query({
        "inputs": user_prompt,
    })
    
    # Generate image from Stable Diffusion
    stable_diffusion_image = pipe(user_prompt, height=512, width=512).images[0]
    
    # Enhance the Stable Diffusion image
    stable_diffusion_image_enhanced = enhance_image(stable_diffusion_image)
    
    # Compute CLIP scores
    flux_image = Image.open(io.BytesIO(flux_image_bytes)) if flux_image_bytes else None
    flux_clip_score = compute_clip_score(flux_image, user_prompt) if flux_image else None
    stable_diffusion_clip_score = compute_clip_score(stable_diffusion_image_enhanced, user_prompt)
    
    # Save or return images
    flux_image_path = "flux_image.png"
    stable_diffusion_image_enhanced_path = "stable_diffusion_image.png"
    
    flux_image.save(flux_image_path) if flux_image else None
    stable_diffusion_image_enhanced.save(stable_diffusion_image_enhanced_path)
    
    return jsonify({
        'flux_image': flux_image_path if flux_image else None,
        'stable_diffusion_image': stable_diffusion_image_enhanced_path,
        'flux_clip_score': flux_clip_score,
        'stable_diffusion_clip_score': stable_diffusion_clip_score
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Get the port from the environment variable, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)