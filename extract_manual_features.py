import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_folder = 'cifar-10-batches-py/'


# Load the train data
for i in range(1, 6):
    data_batch = unpickle(data_folder + 'data_batch_' + str(i))
    if i == 1:
        train_images = data_batch[b'data']
    else:
        train_images = np.vstack((train_images, data_batch[b'data']))

train_images = train_images.reshape(-1, 3, 32, 32)
train_images = train_images.astype(np.uint8)
train_images = np.transpose(train_images, (0, 2, 3, 1))


# Load the test data
data_batch = unpickle(data_folder + 'test_batch')
test_images = data_batch[b'data']
test_images = test_images.reshape(-1, 3, 32, 32)
test_images = test_images.astype(np.uint8)
test_images = np.transpose(test_images, (0, 2, 3, 1))



# Sử dụng các đặc trưng thủ công
def extract_features(image):
    features = {}

    # Chuyển sang grayscale cho các đặc trưng không hỗ trợ RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Thống kê cơ bản trên từng kênh màu
    for i, c in enumerate(['R', 'G', 'B']):
        features[f'mean_{c}'] = np.mean(image[:, :, i])
        features[f'variance_{c}'] = np.var(image[:, :, i])

    # 2. Histogram trên từng kênh màu
    for i, c in enumerate(['R', 'G', 'B']):
        hist, _ = np.histogram(image[:, :, i], bins=16, range=(0, 256))
        hist = hist / np.sum(hist)  # Chuẩn hóa
        for j in range(16):
            features[f'hist_{c}_{j}'] = hist[j]

    # 3. HOG - Histogram of Oriented Gradients trên từng kênh màu
    for i, c in enumerate(['R', 'G', 'B']):
        hog_features, _ = hog(image[:, :, i], pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        for j in range(min(16, len(hog_features))):  # Lấy 16 đặc trưng đầu để giảm chiều
            features[f'hog_{c}_{j}'] = hog_features[j]

    # 4. GLCM - Gray-Level Co-occurrence Matrix
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]

    # 5. Canny Edge Detection (Grayscale)
    edges = cv2.Canny(gray, 100, 200)
    features['edge_mean'] = np.mean(edges)
    features['edge_var'] = np.var(edges)

    # 6. Hu Moments (Grayscale)
    hu_moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    for i in range(7):
        features[f'hu_{i}'] = hu_moments[i]


    return features


scaler = StandardScaler()
batch_size = 16

print("Start extracting features...")

# Extract features for train data
train_features = []
for i in tqdm(range(0, len(train_images), batch_size)):
    batch = train_images[i:i + batch_size]
    batch_features = [extract_features(img) for img in batch]
    train_features.extend(batch_features)
df_train = pd.DataFrame(train_features)
train_features = scaler.fit_transform(df_train)


# Extract features for test data
test_features = []
for i in tqdm(range(0, len(test_images), batch_size)):
    batch = test_images[i:i + batch_size]
    batch_features = [extract_features(img) for img in batch]
    test_features.extend(batch_features)
df_test = pd.DataFrame(test_features)
test_features = scaler.fit_transform(df_test)


# save features
np.save('train_features.npy', train_features)
np.save('test_features.npy', test_features)

print("Features extracted and saved successfully!")