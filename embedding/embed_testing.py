import sys
import os
import matplotlib.pyplot as plt
import numpy as np
#sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path.
from img2vec_pytorch import Img2Vec
from PIL import Image
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity


input_path = './embedding/test_images'
save_folder = './embedding/plots'
max_size = 500


def img_similarity(model: str):
    img2vec = Img2Vec(cuda=True, model=model)
    print("Getting vectors for test images...\n")
    # For each test image, we store the filename and vector as key, value in a dictionary
    pics = {}
    images = []
    for file in os.listdir(input_path):
        filename = os.fsdecode(file)
        images.append(filename)
        img = Image.open(os.path.join(input_path, filename)).convert('RGB')
        vec = img2vec.get_vec(img)
        pics[filename] = vec

    # Create a numpy array to store the cosine similarity values between the images
    cos_sim = np.zeros((len(pics), len(pics)))

    for i, img1 in enumerate(pics.keys()):
        for j, img2 in enumerate(pics.keys()):
            # Calculate the cosine similarity
            cos_sim[i][j] = cosine_similarity(pics[img1].reshape((1, -1)), pics[img2].reshape((1, -1)))[0][0]

    # Create a new figure and axis object
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    cax = ax.matshow(cos_sim, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)

    # Show values inside the table
    for (i, j), z in np.ndenumerate(cos_sim):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    plt.xticks(range(0,len(images)),images, rotation=90)
    plt.yticks(range(0,len(images)),images)
    fig.colorbar(cax)
    plt.title('Cosine Similarity of Images (' + model + ')')
    plt.savefig(os.path.join(save_folder, model + '.png'))

img_similarity('resnet-18')
img_similarity('alexnet')
img_similarity('vgg')
img_similarity('densenet')
img_similarity('efficientnet_b0')
img_similarity('efficientnet_b1')
img_similarity('efficientnet_b2')
img_similarity('efficientnet_b3')
img_similarity('efficientnet_b4')
img_similarity('efficientnet_b5')
img_similarity('efficientnet_b6')
img_similarity('efficientnet_b7')