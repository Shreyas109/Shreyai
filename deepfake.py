import torch
from transformers import AutoModelForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

# Load the pre-trained model from Hugging Face or custom-trained model
checkpoint_path = "E:/project/deep_fake_detector_epoch_15.pth"  # Path to your checkpoint

# Load model and state_dict
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # Set the model to evaluation mode
print(f"Model loaded successfully from {checkpoint_path}")

# Preprocessing function to handle input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (model input size)
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Entropy threshold for declaring as Fake (updated to > 0.2)
entropy_threshold = 0.2

# Function to calculate entropy
def calculate_entropy(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)  # Avoid log(0) error

# Example usage for predictions
image_path = "000023.jpg"  # Replace with your test image path
image_tensor = preprocess_image(image_path)

# Make prediction
with torch.no_grad():  # Disable gradient computation for inference
    outputs = model(image_tensor)  # Forward pass
    logits = outputs.logits  # Logits (raw scores)
    probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
    max_prob, predicted_class = torch.max(probabilities, 1)  # Get predicted class and max probability

# Print out logits, probabilities for debugging
print("Logits: ", logits)
print("Softmax Probabilities: ", probabilities)
print("Predicted Class: ", predicted_class.item())
print("Max Probability: ", max_prob.item())

# Calculate entropy
entropy = calculate_entropy(probabilities)
print("Entropy: ", entropy.item())

# Check if the entropy is above the threshold (indicating high uncertainty)
if entropy.item() > entropy_threshold:
    print("Prediction: Fake (High entropy, uncertain decision)")
else:
    if predicted_class.item() == 0:
        print("Prediction: Real")
    else:
        print("Prediction: Fake")
