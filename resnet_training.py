'''
This script pretrains the Resnet model on the CIFAR-10 dataset (5 train batches) and saves the weights for three different pretraining datasets.
'''

import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from CIFAR10Dataset import pretrain_dataset1, pretrain_dataset2, pretrain_dataset3, val_dataset1, val_dataset2, val_dataset3
from torch.utils.data import DataLoader
from model import *

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

# Dataset groups
datasets = [
    (pretrain_dataset1, val_dataset1, "group_labels1"),
    (pretrain_dataset2, val_dataset2, "group_labels2"),
    (pretrain_dataset3, val_dataset3, "group_labels3")
]

# Training loop for each dataset
for train_dataset, val_dataset, dataset_name in datasets:
    print(f"Training on {dataset_name}...")
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True)

    # Early Stopping Parameters
    patience = 150
    best_val_loss = float("inf")
    early_stop_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs = []

    # Training step
    num_epochs = 1000
    for epoch in range(num_epochs):

        # Train the model
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} ({dataset_name})", leave=False)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            train_accuracy += accuracy

            train_loader_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} ({dataset_name}) | Loss: {loss.item():.4f} | Accuracy: {train_accuracy.item() / len(train_loader):.2f}")
            train_loader_tqdm.update()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy / len(train_loader))
        train_loader_tqdm.close()

        # Evaluate the model on the validation data
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                accuracy = (outputs.argmax(dim=1) == labels).float().mean()
                val_accuracy += accuracy

        print(f"\nEpoch: {epoch + 1}/{num_epochs} ({dataset_name}) | Validation Loss: {val_loss / len(val_loader):.4f} | Accuracy: {val_accuracy.item() / len(val_loader):.2f}\n")

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy / len(val_loader))
        epochs.append(epoch + 1)
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"Resnet_weights_{dataset_name}.pth")
            print(f"✅ Model improved for {dataset_name}, saving...")
        else:
            early_stop_counter += 1
            print(f"Early Stopping Counter ({dataset_name}): {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered for {dataset_name}. Stopping training.")
            break

    val_accuracies = np.array([acc.cpu().numpy() for acc in val_accuracies])
    train_accuracies = np.array([acc.cpu().numpy() for acc in train_accuracies])

    # Plot and save the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss per Epoch ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Resnet_loss_{dataset_name}.png', bbox_inches='tight')
    plt.show()

    # Plot and save the training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training & Validation Accuracy per Epoch ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Resnet_accuracy_{dataset_name}.png', bbox_inches='tight')
    plt.show()
