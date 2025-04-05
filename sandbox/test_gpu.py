#!/usr/bin/env python3
import os

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# Load FashionMNIST dataset without downloading
data_path = "./data"
raw_data_path = os.path.join(data_path, "FashionMNIST/raw")
processed_data_path = os.path.join(data_path, "FashionMNIST/processed")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))],
)

# Ensure dataset files exist before loading
data_files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]
data_available = all(os.path.exists(os.path.join(raw_data_path, f)) for f in data_files)

if not data_available:
    raise RuntimeError(
        "FashionMNIST dataset not found. Ensure extracted files are inside ./data/FashionMNIST/raw/",
    )

# Ensure processed folder exists to prevent torchvision from attempting a re-download
os.makedirs(processed_data_path, exist_ok=True)
open(os.path.join(processed_data_path, "train.pt"), "w").close()
open(os.path.join(processed_data_path, "test.pt"), "w").close()

# Load dataset without downloading
train_dataset = torchvision.datasets.FashionMNIST(
    root=data_path,
    train=True,
    transform=transform,
    download=False,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
)

# Instantiate the model, move to GPU if available
model = SimpleNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")
