import os
import numpy as np
from PIL import Image
from shutil import copyfile
from utils import load_model
from matplotlib import pyplot as plt
# from img2vec_pytorch import Img2Vec
from img_to_vec import Img2Vec
from tqdm import tqdm
import hdbscan
import umap
from torch.utils.data import DataLoader
import pandas as pd
import warnings
from datasets import load_dataset
from torchvision import transforms
import torch
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score
import plotly.express as px
import time
import multiprocessing
from glob import glob
from itertools import islice

img_train_path_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/image_train_paths.pkl'
img_val_path_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/image_val_paths.pkl'
checkpoint = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/checkpoint.txt'
embedding_output_dir = "D:/embeddings"
RETURN_VECTOR_LENGTH = {
    'ResNet': 2048,
    'AlexNet': 4096,
    'VGG': 4096,
    'DenseNet': 1024,
    'EfficientNet': 1792
}
model_name = 'AlexNet'
batch_size = 1024


if __name__ == '__main__':
    # Measure time taken for the whole process
    start_time = time.time()

    def initilazite_model():
        print("Initializing model...")
        # Load the trained model.
        model, input_size = load_model(model_name, feature_extract=True, on_all_data=True)

        img2vec = Img2Vec(model=model,input_size=input_size, cuda=True)
        vec_length = RETURN_VECTOR_LENGTH[model_name] 

        dataloaders_dict = get_dataloaders(input_size=input_size, batch_size=batch_size)
        return img2vec, dataloaders_dict, vec_length
    
    def get_dataloaders(input_size, batch_size):
        print("Initializing dataloaders...")
        # Define the mean and std of the dataset (precomputed)
        mean = torch.tensor([0.5898, 0.5617, 0.5450])
        std = torch.tensor([0.3585, 0.3583, 0.3639])

        # Define the transforms
        transforms = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        
        # Load the datasets
        train_dataset = load_dataset(img_train_path_file, transforms)
        val_dataset = load_dataset(img_val_path_file, transforms)
        image_datasets = {
            'train': train_dataset,
            'val': val_dataset
        }
        # Get the number of CPU cores
        num_workers = multiprocessing.cpu_count()

        # Create the dataloaders
        dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=num_workers-2, shuffle=False) for x in ['train', 'val']}
        return dataloaders_dict
  
    img2vec, dataloaders_dict, vec_length = initilazite_model()

    def load_checkpoint():
        if os.path.exists(checkpoint):
            with open(checkpoint, 'r') as f:
                checkpoint = int(f.read())
            rng_state = torch.load('rng_state.pt')
            torch.set_rng_state(rng_state)
            return checkpoint
        else:
            return 0
    
    print('Loading checkpoint...')
    checkpoint = load_checkpoint()

    print('Reading images...')
    for phase in ['train', 'val']:
        # Get the dataloader for this phase
        dataloader = dataloaders_dict[phase]
        # Skip to the checkpoint if one exists
        if checkpoint > 0:
            dataloader = islice(dataloader, checkpoint, None)
        # Iterate over the batches
        for inputs, labels in dataloader:
            try:                
                # Get the image vectors
                vecs = img2vec.get_vec(inputs, tensor=False)
                # Get the class names from the labels via the dataset
                class_names = [dataloaders_dict[phase].idx_to_class[label] for label in labels]
                # Save the vectors to disk as float16 in this format: cnt_i_class_name.npy
                for i, vec in enumerate(vecs):
                    np.save(os.path.join(embedding_output_dir, f'{checkpoint}_{class_names[i]}.npy'), vec.astype(np.float16))
            except Exception as e:
                print(e)
                print('Error at cnt:', checkpoint)
                with open(checkpoint, 'w') as f:
                    f.write(f'{checkpoint}')
            checkpoint += len(inputs)

################################################################################################################################################################


    filepaths = sorted(glob(os.path.join(embedding_output_dir, '*.npy')))
    ground_truth = [int(fp.split('_')[-2]) for fp in filepaths]
    vecs = np.concatenate([np.load(fp) for fp in tqdm(filepaths)], axis=0)


    print('Preprocessing...')
    reducer = umap.UMAP(n_neighbors=40, min_dist=0.0, n_components=50, random_state=42)
    reduced_feature_vectors = reducer.fit_transform(vecs)

    print('Clustering...')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    cluster_labels = clusterer.fit_predict(reduced_feature_vectors)

    print('Cluster analysis...')
    print("Extrinsic Evaluation:")
    ari = adjusted_rand_score(ground_truth, cluster_labels)
    print("Adjusted Rand Index:", ari)
    ami = adjusted_mutual_info_score(ground_truth, cluster_labels)
    print("Adjusted Mutual Information:", ami)
    fmi = fowlkes_mallows_score(ground_truth, cluster_labels)
    print("Fowlkes-Mallows Index:", fmi)


    print('Reducing dimensions for visualization...')
    reducer = umap.UMAP(n_neighbors=40, min_dist=0.0, n_components=2, random_state=42)
    embedding = reducer.fit_transform(reduced_feature_vectors)

    print('Statistics:')
    # Get the cluster labels from the fitted HDBSCAN object
    cluster_labels = clusterer.labels_

    # Calculate the number of clusters
    num_clusters = len(np.unique(cluster_labels)) - 1  # Subtract 1 to account for the noise cluster (-1)

    # Calculate the number of noise points
    num_noise_points = np.sum(cluster_labels == -1)

    # Calculate the noise percentage
    noise_percentage = (num_noise_points / len(cluster_labels)) * 100

    print(f"Number of clusters formed: {num_clusters}")
    print(f"Number of noise points: {num_noise_points}")
    print(f"Noise percentage: {noise_percentage:.2f}%")

    # print('Plotting...')
    # fig = px.scatter(
    #     x=embedding[:, 0],
    #     y=embedding[:, 1],
    #     color=cluster_labels,
    #     hover_name=image_filenames,
    #     labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': 'Cluster'},
    #     title='UMAP projection of the dataset, colored by HDBSCAN clusters',
    #     color_continuous_scale='Viridis',
    #     width=800,
    #     height=600,
    # )

    fig.show()
    fig.write_html('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/umap.html')
    # Check time taken
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
