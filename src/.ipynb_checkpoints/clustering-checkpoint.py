import os
import numpy as np
from PIL import Image
from shutil import copyfile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import load_model
from matplotlib import pyplot as plt
from datasets import get_dataloaders
from img2vec_pytorch import Img2Vec
from tqdm import tqdm
import hdbscan
import umap
import pandas as pd
import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score
import plotly.express as px
import time
from glob import glob

if __name__ == '__main__':
    # Measure time taken for the whole process
    start_time = time.time()
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    RETURN_VECTOR_LENGTH = {
        'ResNet': 2048,
        'AlexNet': 4096,
        'VGG': 4096,
        'DenseNet': 1024,
        'EfficientNet': 1792
    }

    model_name = 'AlexNet'
    embedding_output_dir = "D:/embeddings"

    # Load the trained model.
    model, input_size = load_model(model_name, feature_extract=True, on_all_data=True)

    img2vec = Img2Vec(model=model,input_size=input_size, cuda=True)
    vec_length = RETURN_VECTOR_LENGTH[model_name] 

    train_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/train_file_counts.csv')
    train_class_counts = {row['class']: row['count'] for row in train_class_counts.to_dict(orient='records')}
    num_classes = len(train_class_counts)

    val_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/val_file_counts.csv')
    val_class_counts = {row['class']: row['count'] for row in val_class_counts.to_dict(orient='records')}

    train_val_class_counts_dict = {
        'train': train_class_counts,
        'val': val_class_counts
    }

    batch_size = 32
    combined_dataloader = get_dataloaders(train_val_class_counts_dict=train_val_class_counts_dict,input_size=input_size, batch_size=batch_size, combined=True)

    samples = len(combined_dataloader.dataset)

    # Matrix to hold the image vectors
    # vec_mat = np.zeros((samples, vec_length))

    print('Reading images...')
    cnt = 0
    for inputs, labels in tqdm(combined_dataloader):
        # Get the image vectors
        vecs = img2vec.get_vec(inputs, tensor=False)
        # Save the vectors to disk as float16
        for i, vec in enumerate(vecs):
            np.save(os.path.join(embedding_output_dir, f'({cnt}_{labels[i]}_{i}.npy'), vec.astype(np.float16))
        cnt += len(vecs)


    filepaths = sorted(glob(os.path.join(embedding_output_dir, '*.npy')))
    vecs = np.concatenate([np.load(fp) for fp in tqdm(filepaths)], axis=0)

    print('Preprocessing...')
    reducer = umap.UMAP(n_neighbors=40, min_dist=0.0, n_components=50, random_state=42)
    reduced_feature_vectors = reducer.fit_transform(vecs)

    print('Clustering...')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    cluster_labels = clusterer.fit_predict(reduced_feature_vectors)

    # print('Cluster analysis...')
    # print("Extrinsic Evaluation:")
    # ari = adjusted_rand_score(ground_truth, cluster_labels)
    # print("Adjusted Rand Index:", ari)
    # ami = adjusted_mutual_info_score(ground_truth, cluster_labels)
    # print("Adjusted Mutual Information:", ami)
    # fmi = fowlkes_mallows_score(ground_truth, cluster_labels)
    # print("Fowlkes-Mallows Index:", fmi)


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
