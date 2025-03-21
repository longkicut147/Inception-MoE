'''
This file contains the CIFAR10Dataset class which is used to load the CIFAR-10 dataset.
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#----------------------------------------------------------------------------------------


# Load the dataset
data_folder = 'cifar-10-batches-py/'


# Load the training data (using 4 training batches of Cifar-10)
for i in range(1, 5):
    data_batch = unpickle(data_folder + 'data_batch_' + str(i))
    if i == 1:
        train_images = data_batch[b'data']
        train_labels = data_batch[b'labels']
    else:
        train_images = np.vstack((train_images, data_batch[b'data']))
        train_labels = np.hstack((train_labels, data_batch[b'labels']))

train_images = train_images.reshape(-1, 3, 32, 32)
train_images = torch.tensor(train_images, dtype=torch.float32) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)


# Load the validation data (using the 5th training batch of Cifar-10)
data_batch = unpickle(data_folder + 'data_batch_5')
val_images = data_batch[b'data']
val_labels = data_batch[b'labels']

val_images = val_images.reshape(-1, 3, 32, 32)
val_images = torch.tensor(val_images, dtype=torch.float32) / 255.0
val_labels = torch.tensor(val_labels, dtype=torch.long)


# Dataloader for train data and validation data
train_dataset = CIFAR10Dataset(train_images, train_labels)
val_dataset = CIFAR10Dataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)