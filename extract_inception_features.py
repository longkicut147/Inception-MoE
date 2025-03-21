'''
This script extracts features from the inception model and uses them to train the classifier. (temporarily using the decision tree)
'''

import torch
from tqdm import tqdm

from CIFAR10Dataset import train_dataset, test_dataset
from torch.utils.data import DataLoader

from model import CNN_Inception
from sklearn.tree import DecisionTreeClassifier



# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# print(images_train.shape)
# print(images_test.shape)
# print(labels_train.shape)
# print(labels_test.shape)
# print(type(images_train))
# print(type(images_test))
# print(type(labels_train))
# print(type(labels_test))


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_inception = CNN_Inception(in_channels=3).to(device)
cnn_inception.load_state_dict(torch.load("inception_model.pth"))
cnn_inception.eval()


# Extract features, using the inception model remove the FC layer
features_train = []
labels_train = []
features_test = []
labels_test = []

with torch.no_grad():
    for batch in tqdm(train_loader, desc="Processing Train Batches"):
        batch = batch[0].to(device)
        batch_features = cnn_inception(batch, extract_features=True)
        features_train.append(batch_features)
        labels_train.append(batch[1])
    features_train = torch.cat(features_train, dim=0)
    labels_train = torch.cat(labels_train, dim=0)

    for batch in tqdm(test_loader, desc="Processing Test Batches"):
        batch = batch[0].to(device)
        batch_features = cnn_inception(batch, extract_features=True)
        features_test.append(batch_features)
        labels_test.append(batch[1])
    features_test = torch.cat(features_test, dim=0)
    labels_test = torch.cat(labels_test, dim=0)



# Convert to numpy array and convert to 2D for decision tree classifier
features_train = features_train.detach().cpu().numpy()
features_test = features_test.detach().cpu().numpy()
features_train = features_train.reshape(features_train.shape[0], -1)
features_test = features_test.reshape(features_test.shape[0], -1)


model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(features_train, labels_train)
accuracy = model.score(features_test, labels_test)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")
