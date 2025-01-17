import torch
from flask import Flask, request, jsonify, Response
from transformers import AutoModelForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import io
import time

app = Flask(__name__)

# Load the pre-trained model from Hugging Face or custom-trained model
checkpoint_path = "E:/project/deep_fake_detector_epoch_15.pth"  # Path to your checkpoint

# Load model and state_dict
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded successfully from {checkpoint_path}")

# Preprocessing function to handle input image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Open image from bytes and convert to RGB
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (model input size)
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to calculate entropy
def calculate_entropy(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)  # Avoid log(0) error

# SSE generator to send progress updates
def generate_progress_updates():
    # Simulate progress from 1% to 100% in steps
    for progress in range(1, 101):
        yield f"data: {progress}\n\n"
        time.sleep(0.05)  # Simulate some processing time

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file from the request
        image_bytes = file.read()

        # Preprocessing the image
        image_tensor = preprocess_image(image_bytes)

        # Start streaming progress
        def progress_stream():
            # Preprocessing progress
            for progress in generate_progress_updates():
                yield progress
            # Model inference progress (simulating)
            for progress in generate_progress_updates():
                yield progress
            # Entropy calculation progress (simulating)
            for progress in generate_progress_updates():
                yield progress

        return Response(progress_stream(), content_type='text/event-stream')

    except Exception as e:
        # Log the error traceback for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
