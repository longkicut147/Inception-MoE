import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CIFAR10Dataset import pretrain_dataset, val_dataset
from model import CNN_Inception, CNN_Inception_Gating, CNN_Resnet, SimpleCNN


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


# Define the Sparse Mixture of Experts Model
class SparseMixtureOfExperts(nn.Module):
    def __init__(self, in_channels, num_experts, expert_models, top_k=2):
        super(SparseMixtureOfExperts, self).__init__()

        self.in_channels = in_channels
        self.num_experts = num_experts
        self.experts = expert_models
        self.top_k = top_k

        self.gating = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_experts),  
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        gate_output = self.gating(x)  # (batch_size, num_experts)
        print(gate_output)
        topk_values, topk_indices = torch.topk(gate_output, self.top_k, dim=1)
        mask = torch.zeros_like(gate_output)
        mask.scatter_(1, topk_indices, 1)
        gate_weights = gate_output * mask
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True)

        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)

        final_output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)

        # print(final_output)
        return final_output



# Data loaders
train_loader = DataLoader(pretrain_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize models and data loaders
num_experts = 3

# inception = CNN_Inception(in_channels=3).to(device)
# inception.load_state_dict(torch.load("weights/Inception_weights.pth"))
# inception.eval()

# simple_cnn = SimpleCNN(in_channels=3).to(device)
# simple_cnn.load_state_dict(torch.load("weights/SimpleCNN_weights.pth"))
# simple_cnn.eval()

resnet1 = CNN_Resnet(in_channels=3).to(device)
resnet1.load_state_dict(torch.load("weights/Resnet_weights_group_labels1.pth"))
resnet2 = CNN_Resnet(in_channels=3).to(device)
resnet2.load_state_dict(torch.load("weights/Resnet_weights_group_labels2.pth"))
resnet3 = CNN_Resnet(in_channels=3).to(device)
resnet3.load_state_dict(torch.load("weights/Resnet_weights_group_labels3.pth"))


experts = [resnet1, resnet2, resnet3]
for expert in experts:
    expert.eval()


model = SparseMixtureOfExperts(in_channels=3, num_experts=num_experts, expert_models=experts).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.gating.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True)


# Early Stopping Parameters
patience = 150  # Số epoch cho phép trước khi dừng
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
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

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

        train_loader_tqdm.set_description(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f} | Accuracy: {train_accuracy.item() / len(train_loader):.2f}")
        train_loader_tqdm.update()

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy / len(train_loader))
    train_loader_tqdm.close()

    # Evaluate the model on the validation data
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    # val_loader = val_loader

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            val_accuracy += accuracy

    print(f"\nEpoch: {epoch + 1}/{num_epochs} | Validation Loss: {val_loss / len(val_loader):.4f} | Accuracy: {val_accuracy.item() / len(val_loader):.2f}\n")

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy / len(val_loader))


    epochs.append(epoch + 1)
    scheduler.step(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "MoE_weights.pth")
        print("✅ Model improved, saving...")
    else:
        early_stop_counter += 1
        print(f"Early Stopping Counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered. Stopping training.")
        break


val_accuracies = np.array([acc.cpu().numpy() for acc in val_accuracies])
train_accuracies = np.array([acc.cpu().numpy() for acc in train_accuracies])


# Plot and save the training loss
# turn back to cpu to plot
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
train_accuracies = np.array(train_accuracies)
val_accuracies = np.array(val_accuracies)
epochs = np.array(epochs)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss per Epoch')
plt.legend()
plt.grid(True)

# Lưu biểu đồ
plt.savefig('MoE_train_val_loss.png', bbox_inches='tight')
plt.show()

# Plot and save the training accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)

# Lưu biểu đồ
plt.savefig('MoE_train_val_accuracy.png', bbox_inches='tight')
plt.show()
