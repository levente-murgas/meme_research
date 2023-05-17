import os
import h5py
from PIL import Image
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

hdf5_file_path = 'D:/Memes2023/dataset.hdf5'
root = "D:/Memes2023_splitted/finetuning"
corrupted = []

def create_hdf5_file(root, hdf5_file_path):
    # Create an HDF5 file to store the dataset
    with h5py.File(hdf5_file_path, 'w') as f:
        for subdir, dirs, files in tqdm(os.walk(root), total=410674):
            for file in files:
                try:
                    # Ensure the file is an image
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        file_path = os.path.join(subdir, file)
                        # Get the class name from the file path
                        class_name = os.path.basename(os.path.dirname(file_path))
                        # Open the image using PIL
                        img = Image.open(file_path)
                        img_arr = np.array(img)
                        # Create a group for each class
                        if class_name not in f:
                            class_group = f.create_group(class_name)
                        else:
                            class_group = f[class_name]
                        # Save the image to the group
                        class_group.create_dataset(file, data=img_arr)
                except:
                    print("Corrupted file: ", file_path)
                    corrupted.append(file_path)
                    continue


create_hdf5_file(root, hdf5_file_path)
# save the corrupted files to a text file
with open('corrupted.txt', 'w') as f:
    for item in corrupted:
        f.write("%s\n" % item)