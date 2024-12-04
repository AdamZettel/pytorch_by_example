import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Custom dataset class to load the images and labels
class AngleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx].unsqueeze(0), self.labels[idx]  # Unsqueeze to add channel dimension

# CNN model for angle prediction
class CNNAnglePredictor(nn.Module):
    def __init__(self):
        super(CNNAnglePredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Adjust based on image size
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten before the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to generate the dataset
def generate_training_data(num_samples=1000, image_size=256, line_thickness=3):
    save_path = Path("training_data")
    save_path.mkdir(parents=True, exist_ok=True)
    
    images = []
    labels = []
    
    for angle_deg in range(0, num_samples):
        angle_rad = np.deg2rad(angle_deg)
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        x_pixel = int((x + 1) / 2 * (image_size - 1))
        y_pixel = int((y + 1) / 2 * (image_size - 1))
        
        img = np.zeros((image_size, image_size), dtype=np.float32)
        origin_x, origin_y = image_size // 2, image_size // 2
        
        for i in range(image_size):
            line_x = int(origin_x + i * (x_pixel - origin_x) / (image_size - 1))
            line_y = int(origin_y + i * (y_pixel - origin_y) / (image_size - 1))
            line_x = max(0, min(image_size - 1, line_x))
            line_y = max(0, min(image_size - 1, line_y))
            
            for dx in range(-line_thickness, line_thickness + 1):
                for dy in range(-line_thickness, line_thickness + 1):
                    nx = max(0, min(image_size - 1, line_x + dx))
                    ny = max(0, min(image_size - 1, line_y + dy))
                    img[ny, nx] = 1.0
        
        images.append(img)
        labels.append(angle_deg)
        
        img_filename = save_path / f"angle_{angle_deg:03d}.png"
        plt.imsave(img_filename, img, cmap='gray', format='png')
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Generate the dataset
images, labels = generate_training_data(num_samples=100, image_size=256, line_thickness=3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create Dataset and DataLoader objects
train_dataset = AngleDataset(X_train, y_train)
test_dataset = AngleDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, loss, and optimizer
model = CNNAnglePredictor()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)  # Squeeze to remove channel dimension
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluation Loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        predicted_angles = outputs.squeeze()
        total += labels.size(0)
        correct += torch.sum(torch.abs(predicted_angles - labels) < 5).item()  # Allow small error
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
