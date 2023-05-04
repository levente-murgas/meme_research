import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
from tqdm.auto import tqdm
import copy

from model import initialize_model
from datasets import train_test_split, get_dataloaders
from utils import save_model, save_plots

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs),desc="Epochs", total=num_epochs, unit="epoch"):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)
            if phase == 'train':
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc, train_loss, val_loss

if __name__ == "__main__":
    # Top level data directory. Here we assume the format of the directory conforms 
    #   to the ImageFolder structure
    data_dir = "./finetuning"

    # Models to choose from [ResNet, AlexNet, VGG, DenseNet, EfficientNet]
    model_name = "AlexNet"

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for 
    num_epochs = 25

    # Flag for feature extracting. When False, we finetune the whole model, 
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print("Start splitting the dataset into train and test sets...")
    # train_test_split()

    train_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/train_file_counts.csv')
    train_class_counts = {row['class']: row['count'] for row in train_class_counts.to_dict(orient='records')}
    num_classes = len(train_class_counts)

    val_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/val_file_counts.csv')
    val_class_counts = {row['class']: row['count'] for row in val_class_counts.to_dict(orient='records')}

    train_val_class_counts_dict = {
        'train': train_class_counts,
        'val': val_class_counts
    }
    print("Initializing the model...")
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    print("Getting the dataloaders...")
    dataloaders_dict = get_dataloaders(train_val_class_counts_dict, input_size=input_size, batch_size=batch_size)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    print("Training the model...")
    # Train and evaluate
    model_ft, train_acc, val_acc, train_loss, val_loss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    # Save the trained model weights.
    save_model(num_epochs, model_ft, optimizer_ft, criterion, feature_extract)
    # Save the loss and accuracy plots.
    save_plots(model_ft, train_acc, val_acc, train_loss, val_loss, feature_extract)
    print('TRAINING COMPLETE')