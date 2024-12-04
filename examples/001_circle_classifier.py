import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

# Step 1: Generate data
np.random.seed(42)
num_points = 5000
radius = 1.0

# Generate random points in 2D
points = np.random.uniform(-1.5, 1.5, size=(num_points, 2))
labels = np.linalg.norm(points, axis=1) <= radius  # Inside circle -> 1, Outside -> 0
labels = labels.astype(np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 2: Define the neural network
class CircleClassifier(nn.Module):
    def __init__(self):
        super(CircleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.fc(x)

# Instantiate the model, define loss and optimizer
model = CircleClassifier()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 3: Train the neural network
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Step 4: Visualize the results
# Generate a grid of points
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# Predict using the trained model
with torch.no_grad():
    grid_preds = model(grid_tensor).numpy().reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 8))
plt.gca().set_aspect('equal')
plt.contourf(xx, yy, grid_preds, levels=50, cmap='RdYlBu', alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolor='k', s=20)
plt.title("Decision Boundary of Neural Network")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label="Probability of Being Inside the Circle")
plt.show()
