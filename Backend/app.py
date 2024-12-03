import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Use the environment variable for the API key
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        return None

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Image Generator API. Use the /generate-image endpoint to generate images."})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    user_prompt = data.get("prompt", "")
    if user_prompt:
        image_bytes = query({"inputs": user_prompt})
        if image_bytes:
            return jsonify({"image": image_bytes.decode("utf-8")})  # Return base64-encoded image
        else:
            return jsonify({"error": "Failed to generate image."}), 400
    return jsonify({"error": "No prompt provided."}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Ensure it uses the correct port
    app.run(host="0.0.0.0", port=port)
