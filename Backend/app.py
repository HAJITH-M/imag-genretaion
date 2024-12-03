import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Use the environment variable for the API key
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}

# Function to query the Hugging Face API
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        print(f"Response Status Code: {response.status_code}")  # Log the status code
        if response.status_code == 200:
            # Encode the image bytes as base64 string
            return base64.b64encode(response.content).decode("utf-8")
        else:
            print(f"Error Response: {response.text}")  # Log the error response body
            return None
    except Exception as e:
        print(f"Error occurred while making request: {str(e)}")  # Log exception
        return None

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Image Generator API. Use the /generate-image endpoint to generate images."})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    user_prompt = data.get("prompt", "")
    
    if not user_prompt:
        return jsonify({"error": "No prompt provided."}), 400

    print(f"Received prompt: {user_prompt}")  # Log the prompt for debugging
    
    # Query the external API to generate the image
    image_base64 = query({"inputs": user_prompt})
    if image_base64:
        return jsonify({"image": image_base64})  # Return base64-encoded image
    else:
        return jsonify({"error": "Failed to generate image."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Ensure it uses the correct port
    app.run(host="0.0.0.0", port=port, debug=True)
