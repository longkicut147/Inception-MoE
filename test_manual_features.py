import numpy as np
from sklearn.tree import DecisionTreeClassifier

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_folder = 'cifar-10-batches-py/'


# Load the train label
for i in range(1, 6):
    data_batch = unpickle(data_folder + 'data_batch_' + str(i))
    if i == 1:
        train_labels = data_batch[b'labels']
    else:
        train_labels = np.hstack((train_labels, data_batch[b'labels']))


# Load the test label
data_batch = unpickle(data_folder + 'test_batch')
test_labels = data_batch[b'labels']
test_labels = np.array(test_labels)



# Load features
features_train = np.load('train_features.npy')
labels_train = train_labels
features_test = np.load('test_features.npy')
labels_test = test_labels

# test with decision tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(features_train, labels_train)
accuracy = model.score(features_test, labels_test)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")