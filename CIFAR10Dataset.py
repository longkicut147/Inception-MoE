'''
This file contains the CIFAR10Dataset class which is used to load the CIFAR-10 dataset.
'''

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import torch
# from torchvision.transforms import v2
import torchvision.transforms as v2


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


transform = v2.Compose([
    v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    # v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#----------------------------------------------------------------------------------------


# Load the CIFAR-10 dataset
data_folder = 'cifar-10-batches-py/'



# Load the pretrain data for models (using 4 training batches of Cifar-10)
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


# Split into groups
mask1 = (pretrain_labels == 0) | (pretrain_labels == 1) | (pretrain_labels == 2) | (pretrain_labels == 3)
mask2 = (pretrain_labels == 4) | (pretrain_labels == 5) | (pretrain_labels == 6)
mask3 = (pretrain_labels == 7) | (pretrain_labels == 8) | (pretrain_labels == 9)

pretrain_images1, pretrain_labels1 = pretrain_images[mask1], pretrain_labels[mask1]
pretrain_images2, pretrain_labels2 = pretrain_images[mask2], pretrain_labels[mask2]
pretrain_images3, pretrain_labels3 = pretrain_images[mask3], pretrain_labels[mask3]

# Augment group images
pretrain_images1_aug = torch.stack([transform(img) for img in pretrain_images1])
pretrain_images2_aug = torch.stack([transform(img) for img in pretrain_images2])
pretrain_images3_aug = torch.stack([transform(img) for img in pretrain_images3])

pretrain_images1 = torch.cat([pretrain_images1, pretrain_images1_aug], dim=0)
pretrain_labels1 = torch.cat([pretrain_labels1, pretrain_labels1], dim=0)
pretrain_images2 = torch.cat([pretrain_images2, pretrain_images2_aug], dim=0)
pretrain_labels2 = torch.cat([pretrain_labels2, pretrain_labels2], dim=0)
pretrain_images3 = torch.cat([pretrain_images3, pretrain_images3_aug], dim=0)
pretrain_labels3 = torch.cat([pretrain_labels3, pretrain_labels3], dim=0)

# Shuffle each group
indices1 = torch.randperm(len(pretrain_labels1))
indices2 = torch.randperm(len(pretrain_labels2))
indices3 = torch.randperm(len(pretrain_labels3))

pretrain_images1, pretrain_labels1 = pretrain_images1[indices1], pretrain_labels1[indices1]
pretrain_images2, pretrain_labels2 = pretrain_images2[indices2], pretrain_labels2[indices2]
pretrain_images3, pretrain_labels3 = pretrain_images3[indices3], pretrain_labels3[indices3]


# Augment and shuffle the all-10-labels pretrain dataset
pretrain_images_aug = torch.stack([transform(img) for img in pretrain_images])
pretrain_images = torch.cat([pretrain_images, pretrain_images_aug], dim=0)
pretrain_labels = torch.cat([pretrain_labels, pretrain_labels], dim=0)
indices = torch.randperm(len(pretrain_labels))
pretrain_images, pretrain_labels = pretrain_images[indices], pretrain_labels[indices]






# Load the validation data for features extraction models (using the 5th training batch of Cifar-10)
val_data_batch = unpickle(data_folder + 'data_batch_5')
val_images = val_data_batch[b'data']
val_labels = val_data_batch[b'labels']

val_images = val_images.reshape(-1, 3, 32, 32)
val_images = torch.tensor(val_images, dtype=torch.float32) / 255.0
val_labels = torch.tensor(val_labels, dtype=torch.long)

# Split into groups
valmask1 = (val_labels == 0) | (val_labels == 1) | (val_labels == 2) | (val_labels == 3)
valmask2 = (val_labels == 4) | (val_labels == 5) | (val_labels == 6)
valmask3 = (val_labels == 7) | (val_labels == 8) | (val_labels == 9)

val_images1, val_labels1 = val_images[valmask1], val_labels[valmask1]
val_images2, val_labels2 = val_images[valmask2], val_labels[valmask2]
val_images3, val_labels3 = val_images[valmask3], val_labels[valmask3]






# Create the datasets
pretrain_dataset = CIFAR10Dataset(pretrain_images, pretrain_labels)
pretrain_dataset1 = CIFAR10Dataset(pretrain_images1, pretrain_labels1)
pretrain_dataset2 = CIFAR10Dataset(pretrain_images2, pretrain_labels2)
pretrain_dataset3 = CIFAR10Dataset(pretrain_images3, pretrain_labels3)

val_dataset = CIFAR10Dataset(val_images, val_labels)
val_dataset1 = CIFAR10Dataset(val_images1, val_labels1)
val_dataset2 = CIFAR10Dataset(val_images2, val_labels2)
val_dataset3 = CIFAR10Dataset(val_images3, val_labels3)





import matplotlib.pyplot as plt
def plot_label_distribution(dataset, title):
    labels = dataset.labels.numpy()
    unique, counts = torch.unique(torch.tensor(labels), return_counts=True)

    plt.bar(unique.numpy(), counts.numpy(), color='skyblue')
    plt.xlabel("Labels")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(unique.numpy())  # Hiển thị tất cả các nhãn trên trục x
    plt.show()

# Vẽ biểu đồ cho từng dataset
plot_label_distribution(pretrain_dataset, "Label Distribution in pretrain_dataset")
plot_label_distribution(pretrain_dataset1, "Label Distribution in pretrain_dataset1")
plot_label_distribution(pretrain_dataset2, "Label Distribution in pretrain_dataset2")
plot_label_distribution(pretrain_dataset3, "Label Distribution in pretrain_dataset3")