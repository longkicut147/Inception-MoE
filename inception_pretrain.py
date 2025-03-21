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
train_loader = DataLoader(pretrain_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)


# --------------------------------------------------------------------------------


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Inception(dropout=0.75).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Early Stopping Parameters
patience = 100  # Số epoch cho phép trước khi dừng
best_val_loss = float("inf")
early_stop_counter = 0

train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        train_accuracy.append(accuracy)

        train_loader_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2f}")
        train_loader_tqdm.update()

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

            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            val_accuracy.append(accuracy)

    print(f"\nValidation Loss: {val_loss / len(val_loader):.4f} | Accuracy: {accuracy.item():.2f}")

    val_losses.append(val_loss / len(val_loader))


    epochs.append(epoch + 1)


    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "Inception_weights.pth")
        print("✅ Model improved, saving...")
    else:
        early_stop_counter += 1
        print(f"Early Stopping Counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered. Stopping training.")
        break


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
