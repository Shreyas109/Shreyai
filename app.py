from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load your pre-trained model
model = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model")

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        image = Image.open(file.stream)
        
        # Process the image (ensure your model is compatible with PIL image format)
        result = model(image)

        # Extract prediction and confidence
        label = result[0]['label']
        confidence = result[0]['score']

        return jsonify({
            'isDeepfake': label.lower() == 'fake',  # Deepfake vs real classification
            'confidenceScore': confidence
        })

    except Exception as e:
        # Log the error traceback for debugging
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error during deepfake detection: {error_message}")
        print(f"Traceback: {traceback_str}")

        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
