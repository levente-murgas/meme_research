import torch
import os
from model import initialize_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.style.use('ggplot')
def save_model(epochs, model, optimizer, criterion, feature_extract):
    """
    Function to save the trained model to disk.
    """
    model_name = model.__class__.__name__
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"C:/Users/Murgi/Documents/GitHub/meme_research/outputs/{model_name}_feature_extract_{feature_extract}.pth")
    
def save_model_2(epochs, model, optimizer, criterion, feature_extract):
    """
    Function to save the trained model to disk. Trained on the combined dataset.
    """
    model_name = model.__class__.__name__
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"C:/Users/Murgi/Documents/GitHub/meme_research/outputs/{model_name}_feature_extract_{feature_extract}_ONALLDATA.pth")
    
def load_model(model_name, feature_extract, on_all_data=True):
    """
    Function to load the saved model from disk.
    """
    train_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/train_file_counts.csv')
    train_class_counts = {row['class']: row['count'] for row in train_class_counts.to_dict(orient='records')}
    num_classes = len(train_class_counts)
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    if on_all_data:
        checkpoint = torch.load(f'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/{model_name}_feature_extract_{feature_extract}_ONALLDATA.pth', map_location='cpu')
    else:
        checkpoint = torch.load(f'C:/Users/Murgi/Documents/GitHub/meme_research/outputs/{model_name}_feature_extract_{feature_extract}.pth', map_location='cpu')
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, input_size
    
def save_plots(model, train_acc, valid_acc, train_loss, valid_loss, feature_extract):
    """
    Function to save the loss and accuracy plots to disk.
    """
    model_name = model.__class__.__name__

    # Convert lists to numpy arrays
    train_acc = [tensor.cpu().numpy() for tensor in train_acc]
    valid_acc = [tensor.cpu().numpy() for tensor in valid_acc]

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"C:/Users/Murgi/Documents/GitHub/meme_research/outputs/accuracy_{model_name}_feature_extract_{feature_extract}.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"C:/Users/Murgi/Documents/GitHub/meme_research/outputs/loss_{model_name}_feature_extract_{feature_extract}.png")

def save_train_plots(model, train_acc, train_loss, feature_extract):
    """
    Function to save the loss and accuracy plots to disk.
    """
    model_name = model.__class__.__name__

    # Convert lists to numpy arrays
    train_acc = [tensor.cpu().numpy() for tensor in train_acc]

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"C:/Users/Murgi/Documents/GitHub/meme_research/outputs/accuracy_{model_name}_feature_extract_{feature_extract}_ONALLDATA.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"C:/Users/Murgi/Documents/GitHub/meme_research/outputs/loss_{model_name}_feature_extract_{feature_extract}_ONALLDATA.png")