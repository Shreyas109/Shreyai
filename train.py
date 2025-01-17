import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from transformers import AutoModelForImageClassification, AutoProcessor
from torchvision import transforms
from PIL import Image
# Path to the dataset folder
image_dir = '/path/to/celeb_df_v2/'

# Define transforms to resize and normalize images
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit the model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load images and labels from the dataset
def load_images_and_labels(image_dir):
    images = []
    labels = []
    for label in ['real', 'fake']:  # Assuming folders 'real' and 'fake'
        label_dir = os.path.join(image_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            img = Image.open(image_path).convert("RGB")
            img = image_transforms(img)
            images.append(img)
            labels.append(0 if label == 'real' else 1)  # 0 for real, 1 for fake
    return torch.stack(images), torch.tensor(labels)

# Load images and labels
images, labels = load_images_and_labels(image_dir)
# Load the pre-trained model and processor
model_name = "prithivMLmods/Deep-Fake-Detector-Model"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
# Create a TensorDataset and DataLoader
dataset = TensorDataset(images, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).logits
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
