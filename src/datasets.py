import glob
import os
import torch
import random
import shutil
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import albumentations as A
from PIL import ImageFile, Image
from torch.utils.data import Dataset


import pickle

train_sampler_file = "C:/Users/Murgi/Documents/GitHub/meme_research/outputs/train_sampler.pkl"
val_sampler_file = "C:/Users/Murgi/Documents/GitHub/meme_research/outputs/val_sampler.pkl"


def save_sampler(sampler, train=True):
    if train:
        with open(train_sampler_file, 'wb') as f:
            pickle.dump(sampler, f)
    else:
        with open(val_sampler_file, 'wb') as f:
            pickle.dump(sampler, f)

def load_sampler(train=True):
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


data_dir = 'D:/Memes2023_splitted/finetuning'

class ImageFolderWithBalancing(datasets.ImageFolder):
    def __init__(self, root, class_img_counts: dict, transform=None, input_size=224, **kwargs):
        super().__init__(root, transform=transform, **kwargs)
        self.class_img_counts = class_img_counts
        self.input_size = input_size
        self.augmented_samples = self._create_augmented_samples()

    def _create_augmented_samples(self):
        augmented_samples = []
        # Calculate the average number of images per class
        num_classes = len(self.class_to_idx)
        total_images = len(self.samples)
        average_images_per_class = total_images // num_classes
        # Create a reverse mapping from indices to class names
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}


        for path, label in self.samples:
            class_name = idx_to_class[label]  # Get the actual class name using the reverse mapping
            current_class_count = self.class_img_counts[class_name]
            if current_class_count < average_images_per_class:
                multiplier = (average_images_per_class - current_class_count) // current_class_count
                # Create additional samples
                for _ in range(multiplier):
                    augmented_samples.append((path, label))
        return augmented_samples

    def __len__(self):
        return len(self.samples) + len(self.augmented_samples)

    def __getitem__(self, index):
        try:
            if index < len(self.samples):
                path, target = self.samples[index]
            else:
                index -= len(self.samples)
                path, target = self.augmented_samples[index]

            sample = self.loader(path)
            resize_transform = transforms.Resize((self.input_size, self.input_size)) # Set the desired size
            sample = resize_transform(sample)

            if self.transform is not None:
                sample = self.transform(sample)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target
        except Exception as e:
            print(f"Error processing image: {path} - {str(e)}")
            return self.__getitem__(index + 1)

def train_test_split():
    src_folder = 'D:/Memes2023/'
    train_folder = 'D:/Memes2023_splitted/finetuning/train/'
    val_folder = 'D:/Memes2023_splitted/finetuning/val/'
    
    with open ('memes.txt', 'r') as f:
        memes = f.readlines()
    for meme in tqdm(memes):
        meme_name = meme.split('/')[-1].strip()
        pattern = src_folder + meme_name + '*'
        files = glob.glob(pattern)
        meme_name = meme_name.replace('%', '')
        if not os.path.exists(train_folder + meme_name) and not os.path.exists(val_folder + meme_name):
            os.mkdir(train_folder + meme_name)
            os.mkdir(val_folder + meme_name)
        else:
            continue
        if len(files) == 1:
            file_name = os.path.basename(files[0])
            shutil.move(files[0], train_folder + meme_name + '/' + file_name)
        elif len(files) > 1:
            random.shuffle(files)
            train = files[:int(len(files)*0.8)]
            val = files[int(len(files)*0.8):]
            for file in train:
                # extract file name form file path
                file_name = os.path.basename(file)
                file_name = file_name.replace('%', '')
                shutil.move(file, train_folder + meme_name + '/' + file_name)
            for file in val:
                # extract file name form file path
                file_name = os.path.basename(file)
                shutil.move(file, val_folder + meme_name + '/' + file_name)

def get_sampler(class_img_counts: dict, train:bool) -> WeightedRandomSampler:
    print("Calculating class weights...")
    #If sampler already exists, return it
    sampler = load_sampler(train)
    if sampler is not None:
        return sampler
    else: #Else, calculate it
        # Calculate the total number of samples
        total_samples = sum(class_img_counts.values())
        # Calculate the weight for each class
        class_weights = {cls: total_samples / count for cls, count in class_img_counts.items()}
        # Create a list of the dataset's labels (replace `labels` with your actual list of labels)
        labels = list(class_img_counts.keys())
        # Create a list of weights for each sample in the dataset based on their class
        sample_weights = [class_weights[label] for label in labels]
        # Create the WeightedRandomSampler
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(labels), replacement=True)
        save_sampler(sampler, train)
        return sampler


def get_dataloaders(train_val_class_counts_dict, input_size=224, batch_size=32)-> dict:
    """Data augmentation and normalization for training
        Just normalization for validation"""
    
    print("Get train sampler...")
    train_sampler = get_sampler(train_val_class_counts_dict['train'], train=True)

    print("Get val sampler...")
    val_sampler = get_sampler(train_val_class_counts_dict['val'], train=False)

    sampler_dict = {
        'train': train_sampler,
        'val': val_sampler
    }

    # print("Getting the mean and std...")
    # mean, std = get_mean_and_std(train_val_class_counts_dict['train'], input_size=input_size, sampler=train_sampler, batch_size=batch_size)
    # print("Mean: {}".format(mean))
    # print("Std: {}".format(std))
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

    # Create training and validation datasets
    image_datasets = {x: ImageFolderWithBalancing(os.path.join(data_dir, x), train_val_class_counts_dict[x], data_transforms[x]) for x in ['train', 'val']}
    ### Combine train and val datasets for training on all data
    combined_samples = image_datasets['train'].samples + image_datasets['val'].samples
    combined_labels = image_datasets['train'].targets + image_datasets['val'].targets
    combined_dataset = CombinedDataset(combined_samples, combined_labels, transform=data_transforms['train'])
    combined_dataloader = DataLoader(combined_dataset, batch_size, sampler=sampler_dict['train'], num_workers=4)
    return combined_dataloader

    # Create training and validation dataloaders
    # dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=4, sampler=sampler_dict[x]) for x in ['train', 'val']}

    return dataloaders_dict

def get_mean_and_std(class_img_counts, sampler, input_size, batch_size=32):
    """Compute the mean and std value of dataset."""
    dataset = ImageFolderWithBalancing(os.path.join(data_dir, 'train'), class_img_counts, transforms.ToTensor(), input_size=input_size)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=False, sampler=sampler)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(dataloader, desc="Calculating mean and std"):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


class CombinedDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)




