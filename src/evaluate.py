import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from utils import load_model
from datasets import get_dataloaders
from tqdm import tqdm

if __name__ == "__main__":
  # Batch size for training (change depending on how much memory you have)
  batch_size = 32

  feature_extract = True

  # Models to choose from [ResNet, AlexNet, VGG, DenseNet, EfficientNet]
  model_name = "AlexNet"

  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  train_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/train_file_counts.csv')
  train_class_counts = {row['class']: row['count'] for row in train_class_counts.to_dict(orient='records')}
  num_classes = len(train_class_counts)

  val_class_counts = pd.read_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/val_file_counts.csv')
  val_class_counts = {row['class']: row['count'] for row in val_class_counts.to_dict(orient='records')}

  train_val_class_counts_dict = {
      'train': train_class_counts,
      'val': val_class_counts
  }

  print('Load the trained model...')
  model, input_size = load_model(model_name, feature_extract=True, on_all_data=True)
  
  # Send the model to GPU
  model = model.to(device)

  print("Get combined dataloader...")
  combined_dataloader = get_dataloaders(train_val_class_counts_dict, input_size=input_size, batch_size=batch_size, combined=True, training=False)

  model.eval() #prep model for evaluation
  # Setup the loss fxn
  criterion = nn.CrossEntropyLoss()

  #track the test loss
  test_loss = 0
  class_correct = {}
  class_total = {}

  for images, labels, class_name in tqdm(combined_dataloader,  total=len(combined_dataloader)):
    images = images.to(device)
    labels = labels.to(device)
    #forward pass 
    output = model(images)
    #calculate the loss
    loss = criterion(output, labels)
    #update the test loss
    test_loss += loss.item()*images.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    #compare predictions to the true labes
    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    #calculate test accuracy for each object class
    for i in range(len(labels)):
      cls = class_name[i]
      label = labels.data[i]
      #create entry for class if it has't been created yet
      if cls not in class_correct:
        class_correct[cls] = 0
        class_total[cls] = 0
      class_correct[cls] += correct[i].item()
      class_total[cls] +=1

  #calcaulate and prınt test loss
  test_loss = test_loss/len(combined_dataloader.sampler)
  print('Test Loss: {:.6f}\n'.format(test_loss))

  print(f'\nTest Accuracy (Overall): {100. * np.sum(list(class_correct.values())) / np.sum(list(class_total.values()))} ({np.sum(class_correct.values)}/{np.sum(class_total.values)}')

  class_accuracies = {k: (100 * v / class_total[k]) for k, v in class_correct.items() if class_total[k] > 0}
  names = list(class_accuracies.keys())
  values = list(class_accuracies.values())

  # Save class accuracies to a csv
  df = pd.DataFrame({'class': names, 'accuracy': values})
  df.to_csv('C:/Users/Murgi/Documents/GitHub/meme_research/outputs/plots/class_accuracies.csv', index=False)

  plt.figure(figsize=(20,10))  # adjust as needed
  plt.bar(names, values)
  plt.xlabel('Classes')
  plt.ylabel('Accuracy (%)')
  plt.title('Test Accuracy of Each Class')
  plt.xticks(rotation=90)  # rotate x labels for better readability if class names are long
  plt.show()