'''
This script pretrains the Inception model on the CIFAR-10 dataset (5 train batches) and save the weights.
'''

import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from CIFAR10Dataset import pretrain_dataset, val_dataset
from torch.utils.data import DataLoader
from model import CNN_Inception


# Set the seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Data loaders
train_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# --------------------------------------------------------------------------------


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Inception().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_losses = []
val_losses = []
epochs = []


# Training step
num_epochs = 100
for epoch in range(num_epochs):

    # Train the model
    model.train()
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_loader_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}")
        train_loader_tqdm.update()
    
    print(f"\nEpoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}")

    train_losses.append(train_loss / len(train_loader))


    # Evaluate the model on the validation data
    model.eval()
    val_loss = 0.0
    # val_loader = val_loader

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader))


    # Next epoch
    epochs.append(epoch + 1)


# Save the model
torch.save(model.state_dict(), "Inception_weights.pth")


# Plot and save the training loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss per Epoch')
plt.legend()
plt.grid(True)

# Lưu biểu đồ
plt.savefig('Inception_train_val_loss.png', bbox_inches='tight')
plt.show()
