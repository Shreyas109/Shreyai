import torch
import torch.nn as nn
import os

# Define the model class (simple CNN for illustration)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 222 * 222, 2)  # Adjust based on input size and architecture
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

# Initialize the model
model = SimpleCNN()

# Dummy input for training (simulate input data)
dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image

# Dummy training loop (you should replace this with your actual training code)
output = model(dummy_input)
# Normally here you'd run your training loop, backpropagate, and update weights.

# After training, save the model
save_dir = 'E:/project/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the model's state_dict
model_save_path = os.path.join(save_dir, 'deepfake_detector.pth')
torch.save(model.state_dict(), model_save_path)

print(f"Model saved successfully at {model_save_path}")


