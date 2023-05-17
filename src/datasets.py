import glob
import os
import torch
import random
import shutil
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler, BatchSampler, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import ImageFile, Image
from torch.utils.data import Dataset
import h5py
from PIL import ImageFile
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict
import multiprocessing

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



class ImageFolderWithBalancing(datasets.ImageFolder):
    def __init__(self, root, class_img_counts: dict, transform=None, input_size=224, **kwargs):
        super().__init__(root, transform=transform, **kwargs)
        self.class_img_counts = class_img_counts
        self.input_size = input_size

        if os.path.exists(augmented_samples_file):
            with open(augmented_samples_file, "rb") as f:
                self.augmented_samples = pickle.load(f)
        else:
            self.augmented_samples = self._create_augmented_samples()
            with open(augmented_samples_file, "wb") as f:
                pickle.dump(self.augmented_samples, f)

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
        
    def save_metadata(self, file_path):
        metadata = {
            'samples': self.samples,
            'targets': self.targets,
            'augmented_samples': self.augmented_samples,
            'class_to_idx': self.class_to_idx
        }
        with open(file_path, 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_from_metadata(cls, file_path, root, class_img_counts, transform=None, input_size=224, **kwargs):
        with open(file_path, 'rb') as f:
            metadata = pickle.load(f)

        instance = cls(root, class_img_counts, transform=transform, **kwargs)
        instance.samples = metadata['samples']
        instance.targets = metadata['targets']
        instance.augmented_samples = metadata['augmented_samples']
        instance.class_to_idx = metadata['class_to_idx']
        return instance


# To load the datasets:
def load_dataset(cache_path, transform):
    with open(cache_path, 'rb') as f:
        img_paths = pickle.load(f)
    return CachedImageFolder(img_paths, transform=transform)

def train_test_split_memes():
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

def get_sampler(class_img_counts: dict, train:bool, combined:bool=False) -> WeightedRandomSampler:
    print("Calculating class weights...")
    #If sampler already exists, return it
    sampler = load_sampler(train, combined)
    if sampler is not None:
        return sampler
    else: #Else, calculate it
        if combined:
            class_img_counts = { k: class_img_counts['train'].get(k, 0) + class_img_counts['val'].get(k, 0) for k in set(class_img_counts['train']) | set(class_img_counts['val']) }
        # Calculate the total number of samples
        total_samples = sum(class_img_counts.values())
        print(f"Total samples: {total_samples}")
        # Calculate the weight for each class
        class_weights = {cls: total_samples / count for cls, count in class_img_counts.items()}
        # Create a list of the dataset's labels (replace `labels` with your actual list of labels)
        labels = list(class_img_counts.keys())
        # Create a list of weights for each sample in the dataset based on their class
        sample_weights = [class_weights[label] for label in labels]
        # Create the WeightedRandomSampler
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=total_samples, replacement=True)
        save_sampler(sampler, train, combined)
        return sampler


def get_dataloaders(train_val_class_counts_dict, input_size=224, batch_size=32, combined=False, training=True):
    """Data augmentation and normalization for training
        Just normalization for validation"""
    if training:
        if combined:
            print("Get combined sampler...")
            combined_sampler = get_sampler(train_val_class_counts_dict, train=True, combined=combined)
        else:
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
    start_time = time.time()

    # if os.path.exists(train_dataset_metadata):
    #     train_dataset = ImageFolderWithBalancing.load_from_metadata(train_dataset_metadata, os.path.join(data_dir, 'train'), train_val_class_counts_dict['train'], data_transforms['train'])
    # else:
    #     train_dataset = ImageFolderWithBalancing(os.path.join(data_dir, 'train'), train_val_class_counts_dict['train'], data_transforms['train'])
    #     train_dataset.save_metadata(train_dataset_metadata)

    # if os.path.exists(val_dataset_metadata):
    #     val_dataset = ImageFolderWithBalancing.load_from_metadata(val_dataset_metadata, os.path.join(data_dir, 'val'), train_val_class_counts_dict['val'], data_transforms['val'])
    # else:
    #     val_dataset = ImageFolderWithBalancing(os.path.join(data_dir, 'val'), train_val_class_counts_dict['val'], data_transforms['val'])
    #     val_dataset.save_metadata(val_dataset_metadata)
    #     
    # image_datasets = {
    #     'train': train_dataset,
    #     'val': val_dataset
    # }
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    train_dataset = load_dataset(img_train_path_file, data_transforms['train'])
    val_dataset = load_dataset(img_val_path_file, data_transforms['val'])
    image_datasets = {
        'train': train_dataset,
        'val': val_dataset
    }

    # Get the number of CPU cores
    num_workers = multiprocessing.cpu_count()

    if combined:
        print("Combine train and val datasets for training on all data")
        a1 = image_datasets['train'].samples
        a2 = image_datasets['val'].samples
        combined_samples = image_datasets['train'].samples + image_datasets['val'].samples
        combined_labels = image_datasets['train'].targets + image_datasets['val'].targets
        combined_dataset = CombinedDataset(combined_samples, combined_labels, transform=data_transforms['train'])
        combined_dataloader = DataLoader(combined_dataset, batch_size)
        
        print("Length of train_dataset:", len(image_datasets['train']))
        print("Length of val_dataset:", len(image_datasets['val']))
        print("Length of train_dataset samples:", len(image_datasets['train'].samples))
        print("Length of train_dataset targets:", len(image_datasets['train'].targets))
        print("Length of val_dataset samples:", len(image_datasets['val'].samples))
        print("Length of val_dataset targets:", len(image_datasets['val'].targets))
        print("Length of combined_samples:", len(combined_samples))
        print("Length of combined_labels:", len(combined_labels))
        print("Total samples in the dataset:", len(combined_dataloader.dataset))
        print("Length of the dataloader:", len(combined_dataloader))

        return combined_dataloader
    else:
        # Create training and validation dataloaders
        if training:
            dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=num_workers-2, sampler=sampler_dict[x]) for x in ['train', 'val']}
        else:
            dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, num_workers=num_workers-2) for x in ['train', 'val']}
        print("Time took for getting the dataloaders: {:.2f} seconds".format(time.time() - start_time))
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


def get_dataloaders_hdf5(batch_size, input_size=224):
    """Get dataloaders from hdf5 file"""
    start_time = time.time()
    print("Initializing Datasets and Dataloaders...")
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
    file = h5py.File('D:/Memes2023/dataset.hdf5', 'r')
    #Get all paths to files in the dataset
    classes = list(file.keys())
    image_paths = []
    if os.path.exists(image_paths_file):
        with open(image_paths_file, 'rb') as f:
            image_paths = pickle.load(f)
    else:
        for cls in tqdm(classes, desc="Getting image paths", total=len(classes)):
            image_paths.extend([(cls, img) for img in file[cls].keys()])
        with open(image_paths_file, 'wb') as f:
            pickle.dump(image_paths, f)

    #Split into train and validation set
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    #Create datasets
    train_dataset = HDF5Dataset('D:/Memes2023/dataset.hdf5', image_paths=train_paths, classes=classes, transform = data_transforms['train'])
    val_dataset = HDF5Dataset('D:/Memes2023/dataset.hdf5', image_paths=val_paths, classes=classes, transform = data_transforms['val'])
    #Create dataloaders
    train_loader = fast_loader(train_dataset, batch_size)
    val_loader = fast_loader(val_dataset, batch_size)

    dataloaders_dict = {'train': train_loader, 'val': val_loader}
    print(f"Finished initializing Datasets and Dataloaders in {time.time() - start_time} seconds")
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

class HDF5Dataset(Dataset):
    def __init__(self, file_path, image_paths, classes, transform=None):
        self.file_path = file_path
        self.image_paths = image_paths
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self._load_batch(idx)
        else:
            return self._load_item(idx)

    def _load_item(self, idx):
        class_name, image_name = self.image_paths[idx]
        with h5py.File(self.file_path, 'r') as file:
            img_data = file[f"{class_name}/{image_name}"][:]
        img = Image.fromarray(img_data).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label

    def _load_batch(self, indices):
        class_dict = defaultdict(list)
        for idx in indices:
            class_name, image_name = self.image_paths[idx]
            class_dict[class_name].append(image_name)
        
        imgs, labels = [], []
        with h5py.File(self.file_path, 'r') as file:
            for class_name in class_dict:
                for image_name in class_dict[class_name]:
                    img_data = file[f"{class_name}/{image_name}"][:]
                    img = Image.fromarray(img_data).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    label = self.class_to_idx[class_name]
                    imgs.append(img)
                    labels.append(label)

        return torch.stack(imgs), torch.tensor(labels)


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'

    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)

def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):

    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.

    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )

class CachedImageFolder(Dataset):
    def __init__(self, img_label_pairs, transform=None):
        self.img_label_pairs = img_label_pairs
        self.transform = transform
        unique_labels = sorted(set(label for _, label in self.img_label_pairs))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(unique_labels)}

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