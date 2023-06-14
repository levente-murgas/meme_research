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
from datasets import get_dataloaders, get_dataloaders_hdf5
from utils import save_plots, save_model, load_model_for_training
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_model(model, dataloader, criterion, optimizer, num_epochs=25, start_epoch=None):
    training_start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    if start_epoch is not None:
        rangefor = range(start_epoch, start_epoch+num_epochs)
    else:
        rangefor = range(num_epochs)

    for epoch in tqdm(rangefor,desc="Epochs", total=num_epochs, unit="epoch", mininterval=60):
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

            cnt = 0
            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                # Start timing
                start_time = time.time()

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    forward_start = time.time()
                    outputs = model(inputs)
                    forward_end = time.time()

                    loss_start = time.time()
                    loss = criterion(outputs, labels)
                    loss_end = time.time()
                    
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        opt_start = time.time()
                        loss.backward()
                        optimizer.step()
                        opt_end = time.time()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                forward_time = forward_end - forward_start
                loss_time = loss_end - loss_start
                opt_time = opt_end - opt_start
                
                total_time = time.time() - start_time
                data_loading_time = total_time - forward_time - loss_time - opt_time

                cnt += 1
                print(f"Batch {cnt}/{number_of_batches}")
                if cnt % 100 == 0:
                    print(f"Data loading time: {data_loading_time:.4f}s")
                    print(f"Forward time: {forward_time:.4f}s")
                    print(f"Loss time: {loss_time:.4f}s")
                    print(f"Optimization time: {opt_time:.4f}s")
                    print(f"Total time: {total_time:.4f}s")
                    print('Data loading took this percent of the total time: {:.2f}%'.format(data_loading_time/total_time*100))

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
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
    print(f"Best epoch: {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Training took {time.time() - training_start_time:.2f}s")
    return model, train_acc, train_loss, val_acc, val_loss

if __name__ == "__main__":
    # Top level data directory. Here we assume the format of the directory conforms 
    #   to the ImageFolder structure
    data_dir = "./finetuning"

    # Models to choose from [ResNet, AlexNet, VGG, DenseNet, EfficientNet]
    model_name = "AlexNet"

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    number_of_batches = 410671 // batch_size
    # Number of epochs to train for 
    num_epochs = 10

    # Flag for feature extracting. When False, we finetune the whole model, 
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    continue_training = False
    # print("Start splitting the dataset into train and test sets...")
    # train_test_split()

    train_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/tables/train_file_counts.csv')
    train_class_counts = {row['Class']: row['Count'] for row in train_class_counts.to_dict(orient='records')}
    num_classes = len(train_class_counts)


    print("Initializing the model...")
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # print("Getting the dataloaders...")
    dataloaders_dict = get_dataloaders(input_size=input_size, batch_size=batch_size, training=True)
    # print("Get combined dataloader...")
    # combined_dataloader = get_dataloaders(train_val_class_counts_dict, input_size=input_size, batch_size=batch_size, combined=True)

    # dataloaders_dict = get_dataloaders_hdf5(batch_size=batch_size, input_size=input_size)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    model_ft = model_ft.to(device)

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
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    print("Training the model...")
    if continue_training:
        file_path = "C:/Users/Murgi/Documents/GitHub/meme_research/outputs/AlexNet_feature_extract_True.pth"
        model_ft, optimizer_ft, epoch, loss = load_model_for_training(file_path, model_ft, optimizer_ft)
        # Train and evaluate
        model_ft, train_acc, train_loss, val_acc, val_loss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, start_epoch=epoch)
    else:
        # Train and evaluate
        model_ft, train_acc, train_loss, val_acc, val_loss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    # Save the trained model weights.
    # save_model(num_epochs, model_ft, optimizer_ft, criterion, feature_extract)
    save_model(num_epochs, model_ft, optimizer_ft, criterion, feature_extract, continue_training=continue_training)
    # Save the loss and accuracy plots.
    save_plots(model_ft, train_acc, val_acc, train_loss, val_loss, feature_extract, continue_training=continue_training)
    # save_train_plots(model_ft, train_acc, train_loss, feature_extract)
    print('TRAINING COMPLETE')