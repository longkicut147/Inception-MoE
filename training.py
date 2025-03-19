from CIFAR10Dataset import train_dataset, train_loader
from inception import CNN_Inception

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Inception().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_losses = []
epochs = []

# Training step
num_epochs = 100
for epoch in range(num_epochs):

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

    train_losses.append(train_loss / len(train_loader))
    epochs.append(epoch + 1)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "inception_model.pth")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Training Accuracy: {100 * correct / total:.2f}%")