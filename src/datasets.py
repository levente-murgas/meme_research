import os
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms
from PIL import  Image
from torch.utils.data import Dataset
import time
from sklearn.model_selection import train_test_split
import multiprocessing
from collections import Counter
import pickle

train_sampler_file = "C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/train_sampler.pkl"
val_sampler_file = "C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/val_sampler.pkl"
data_dir = 'D:/Memes2023_splitted/finetuning'
train_dataset_metadata = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/train_dataset.pkl'
val_dataset_metadata = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/val_dataset.pkl'
augmented_samples_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/augmented_samples.pkl'
combined_sampler_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/combined_sampler.pkl'
image_paths_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/image_paths.pkl'
img_train_path_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/image_train_paths.pkl'
img_val_path_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/image_val_paths.pkl'
eval_train_path_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/eval_train_paths.pkl'
eval_val_path_file = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/eval_val_paths.pkl'
CLASS_NAMES_PATH_FILE = 'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/class_names.txt'

def save_classnames(output_path, dataset):
   with open(output_path, 'w') as f:
       for class_name in dataset.class_to_idx:
           f.write(class_name + '\n')

def save_sampler(sampler, train=True, combined=False):
    if combined:
        with open(combined_sampler_file, 'wb') as f:
            pickle.dump(sampler, f)
    else:
        if train:
            with open(train_sampler_file, 'wb') as f:
                pickle.dump(sampler, f)
        else:
            with open(val_sampler_file, 'wb') as f:
                pickle.dump(sampler, f)

def load_sampler(train=True, combined=False):
    if combined:
        if os.path.exists(combined_sampler_file):
            with open(combined_sampler_file, 'rb') as f:
                return pickle.load(f)
        else: 
            return None
    else:
        if train:
            if os.path.exists(train_sampler_file):
                with open(train_sampler_file, 'rb') as f:
                    return pickle.load(f)
            else: 
                return None
        else: # val
            if os.path.exists(val_sampler_file):
                with open(val_sampler_file, 'rb') as f:
                    return pickle.load(f)
            else:
                return None

# To load the datasets:
def load_dataset(cache_path, transform):
    with open(cache_path, 'rb') as f:
        img_paths = pickle.load(f)
    return CachedImageFolder(img_paths, transform=transform)
    
def get_sampler(img_label_pairs) -> WeightedRandomSampler:
    print("Calculating class weights...")
     #Else, calculate it
    # Calculate the total number of samples
    total_samples = len(img_label_pairs)
    print(f"Total samples: {total_samples}")
    # Extract class names from image-label pairs
    class_names = [label for _, label in img_label_pairs]
    # Count the number of samples in each class
    class_counts = Counter(class_names)
    # Calculate the weight for each class
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    # Create a list of the dataset's labels (replace `labels` with your actual list of labels)
    labels = list(class_counts.keys())
    # Create a list of weights for each sample in the dataset based on their class
    sample_weights = [class_weights[label] for label in class_names]
    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=total_samples, replacement=True)
    return sampler
    
def get_dataloaders(input_size=224, batch_size=32, combined=False, training=True):
    """Initializes and returns the train and val dataloaders"""

    mean = torch.tensor([0.5898, 0.5617, 0.5450])
    std = torch.tensor([0.3585, 0.3583, 0.3639])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    print("Initializing Datasets and Dataloaders...")
    start_time = time.time()

    train_image_paths = []
    with open(img_train_path_file, 'rb') as f:
        train_image_paths = pickle.load(f)

    val_image_paths = []
    with open(img_val_path_file, 'rb') as f:
        val_image_paths = pickle.load(f)

    # # Merge train and val image paths
    # img_paths = train_image_paths + val_image_paths
    # # Apply train test split
    # train_image_paths, val_image_paths = train_test_split(img_paths, test_size=0.05, random_state=42, shuffle=True)

    # # Save train and val image paths for evaluation
    # with open(eval_train_path_file, 'wb') as f:
    #     pickle.dump(train_image_paths, f)

    # with open(eval_val_path_file, 'wb') as f:
    #     pickle.dump(val_image_paths, f)

    train_dataset = CachedImageFolder(train_image_paths, transform=data_transforms['train'])
    val_dataset = CachedImageFolder(val_image_paths, transform=data_transforms['val'])

    if not os.path.exists(CLASS_NAMES_PATH_FILE):
        save_classnames(CLASS_NAMES_PATH_FILE, train_dataset)

    image_datasets = {
        'train': train_dataset,
        'val': val_dataset
    }

    if training:
        print("Get train sampler...")
        train_sampler = get_sampler(train_image_paths)

        print("Get val sampler...")
        val_sampler = get_sampler(val_image_paths)
        sampler_dict = {
            'train': train_sampler,
            'val': val_sampler
        }

    # Get the number of CPU cores
    num_workers = multiprocessing.cpu_count()

    if combined:
        print("Combine train and val datasets for training on all data")
        combined_samples = image_datasets['train'].samples + image_datasets['val'].samples
        combined_labels = image_datasets['train'].targets + image_datasets['val'].targets
        combined_dataset = CombinedDataset(combined_samples, combined_labels, transform=data_transforms['train'])
        combined_dataloader = DataLoader(combined_dataset, batch_size)
        return combined_dataloader
    else:
        # Create training and validation dataloaders
        if training:
            dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=num_workers-2, sampler=sampler_dict[x]) for x in ['train', 'val']}
        else:
            dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=num_workers-2) for x in ['train', 'val']}
        print("Time took for getting the dataloaders: {:.2f} seconds".format(time.time() - start_time))
        return dataloaders_dict

class CombinedDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        try:
            path, label = self.samples[index]
            image = Image.open(path).convert('RGB')
            #Extract class name from path
            #Just for testing purposes
            class_name = path.split('\\')[-2]

            if self.transform is not None:
                image = self.transform(image)

            return image, label, class_name
        except Exception as e:
            print(f"Error processing image: {path} - {str(e)}")
            return self.__getitem__(index + 1)
        
    def __len__(self):
        return len(self.samples)  

class CachedImageFolder(Dataset):
    def __init__(self, img_label_pairs, transform=None):
        self.img_label_pairs = img_label_pairs
        self.transform = transform
        unique_labels = sorted(set(label for _, label in self.img_label_pairs))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(unique_labels)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(unique_labels)}

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        try:
            img_path, class_name = self.img_label_pairs[idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = self.class_to_idx[class_name]
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing image: {img_path} - {str(e)}")
            return self.__getitem__(idx + 1)