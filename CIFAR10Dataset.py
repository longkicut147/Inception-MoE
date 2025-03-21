'''
This file contains the CIFAR10Dataset class which is used to load the CIFAR-10 dataset.
'''

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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


# Load the CIFAR-10 dataset
data_folder = 'cifar-10-batches-py/'



# Load the pretrain data for features extraction models (using 4 training batches of Cifar-10)
for i in range(1, 5):
    pretrain_data_batch = unpickle(data_folder + 'data_batch_' + str(i))
    if i == 1:
        pretrain_images = pretrain_data_batch[b'data']
        pretrain_labels = pretrain_data_batch[b'labels']
    else:
        pretrain_images = np.vstack((pretrain_images, pretrain_data_batch[b'data']))
        pretrain_labels = np.hstack((pretrain_labels, pretrain_data_batch[b'labels']))

pretrain_images = pretrain_images.reshape(-1, 3, 32, 32)
pretrain_images = torch.tensor(pretrain_images, dtype=torch.float32) / 255.0
pretrain_labels = torch.tensor(pretrain_labels, dtype=torch.long)


# Load the validation data for features extraction models (using the 5th training batch of Cifar-10)
val_data_batch = unpickle(data_folder + 'data_batch_5')
val_images = val_data_batch[b'data']
val_labels = val_data_batch[b'labels']

val_images = val_images.reshape(-1, 3, 32, 32)
val_images = torch.tensor(val_images, dtype=torch.float32) / 255.0
val_labels = torch.tensor(val_labels, dtype=torch.long)


# Load the data for the classifier models (using test batch of Cifar-10)
data_batch = unpickle(data_folder + 'test_batch')
images = data_batch[b'data']
labels = data_batch[b'labels']

images = images.reshape(-1, 3, 32, 32)
images = torch.tensor(images, dtype=torch.float32) / 255.0
labels = torch.tensor(labels, dtype=torch.long)

# train test split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)



# Create the datasets
pretrain_dataset = CIFAR10Dataset(pretrain_images, pretrain_labels)
val_dataset = CIFAR10Dataset(val_images, val_labels)
train_dataset = CIFAR10Dataset(train_images, train_labels)
test_dataset = CIFAR10Dataset(test_images, test_labels)