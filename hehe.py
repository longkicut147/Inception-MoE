import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm

from inception import CNN_Inception
from CIFAR10Dataset import unpickle

# Load the test data
data_folder = 'cifar-10-batches-py/test_batch'
data_batch = unpickle(data_folder)
images = data_batch[b'data']
labels = data_batch[b'labels']
images = images.reshape(-1, 3, 32, 32)
images = torch.tensor(images, dtype=torch.float32) / 255.0
labels = torch.tensor(labels, dtype=torch.long)

# train-test split
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_inception = CNN_Inception(in_channels=3).to(device)
cnn_inception.load_state_dict(torch.load("inception_model.pth"))
cnn_inception.eval()


# Extract features
features_train = []
batch_size = 64
for i in tqdm(range(0, len(images_train), batch_size), desc="Processing Batches"):
    batch = images_train[i:i + batch_size]
    batch_features = cnn_inception(batch)
    features_train.append(batch_features)
features_train = torch.cat(features_train, dim=0)


features_test = []
for i in tqdm(range(0, len(images_test), batch_size), desc="Processing Batches"):
    batch = images_test[i:i + batch_size]
    batch_features = cnn_inception(batch)
    features_test.append(batch_features)
features_test = torch.cat(features_test, dim=0)


# Convert to numpy array and convert to 2D for decision tree classifier
features_train = features_train.detach().cpu().numpy()
features_test = features_test.detach().cpu().numpy()
features_train = features_train.reshape(features_train.shape[0], -1)
features_test = features_test.reshape(features_test.shape[0], -1)


model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(features_train, labels_train)
accuracy = model.score(features_test, labels_test)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")