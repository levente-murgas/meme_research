import os
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from shutil import copyfile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import load_model
from matplotlib import pyplot as plt
from datasets import get_class_weights, get_dataloaders

RETURN_VECTOR_LENGTH = {
    'ResNet': 2048,
    'AlexNet': 4096,
    'VGG': 4096,
    'DenseNet': 1024,
    'EfficientNet': 1792
}

model_name = 'EfficientNet'

input_path = './finetuning/val/'
files = os.listdir(input_path)
files += os.listdir('./finetuning/train/')

# Load the trained model.
model, input_size = load_model(model_name, True)

img2vec = Img2Vec(model=model,input_size=input_size, cuda=True)
vec_length = RETURN_VECTOR_LENGTH[model_name] 

sampler = get_class_weights()
###TODO: Get mean and std from the dataset
mean = 0
std = 0
batch_size = 32
dataloaders_dict = get_dataloaders(mean=mean, std=std, sampler=sampler, input_size=input_size, batch_size=batch_size)

samples = len(dataloaders_dict['val']) + len(dataloaders_dict['train'])
k_value = len(os.listdir('./finetuning/train'))  # How many clusters

# Matrix to hold the image vectors
vec_mat = np.zeros((samples, vec_length))

print('Reading images...')
cnt = 0
for phase in ['train', 'val']:
    for inputs, _ in dataloaders_dict[phase]:
        imgs = inputs[0].squeeze()
        for img in imgs:
            vec = img2vec.get_vec(img)
            vec_mat[cnt, :] = vec
            cnt += 1


print('Applying PCA...')
reduced_data = PCA(n_components=2).fit_transform(vec_mat)
kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10)
kmeans.fit(reduced_data)

print('Predicting...')
preds = kmeans.predict(reduced_data)

print('Plotting...')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=preds)
plt.show()

print('Done!')