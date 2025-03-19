import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_folder = 'cifar-10-batches-py/'

for i in range(1, 6):
    data_batch = unpickle(data_folder + 'data_batch_' + str(i))
    if i == 1:
        images = data_batch[b'data']
        labels = data_batch[b'labels']
    else:
        images = np.vstack((images, data_batch[b'data']))
        labels = np.hstack((labels, data_batch[b'labels']))

# print(images.shape)
# print(labels.shape)
# print(type(images))
# print(type(labels))

images = images.reshape(-1, 3, 32, 32)
images = torch.tensor(images, dtype=torch.float32) / 255.0
labels = torch.tensor(labels, dtype=torch.long)


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
train_dataset = CIFAR10Dataset(images, labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)