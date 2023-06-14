import torch
from PIL import Image
import torchvision.transforms as transforms
from model import initialize_model
from utils import load_model
import os
import pandas as pd
from tqdm import tqdm

def load_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = f.readlines()

    return [name.strip() for name in class_names]

def preprocess_input(image_path, input_size):
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5898, 0.5617, 0.5450], [0.3585, 0.3583, 0.3639])  # Normalize using ImageNet mean and std
    ])

    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    return input_batch

def predict(model, input_batch, class_names):
    # Move the input to GPU, if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the index of the class with maximum probability
    _, predicted_idx = torch.max(output, 1)
    
    # Get the name of the class
    predicted_class = class_names[predicted_idx.item()]

    return predicted_class, probabilities[predicted_idx.item()].item()

if __name__ == "__main__":
    # Set these to your paths
    MODEL = "AlexNet"
    CLASS_NAMES_PATH = "C:/Users/Murgi/Documents/GitHub/meme_research/outputs/cache/class_names.txt"
    DATASET_PATH = "D:/Memes2022Final2_resized/"

    # Load model
    model, input_size = load_model(MODEL, feature_extract=True, use_continued_train=True)

    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)

    data =[]
    cnt = 0
    for image in tqdm(os.listdir(DATASET_PATH), total=len(os.listdir(DATASET_PATH))):
        if cnt == 1000:
            break
        image_path = os.path.join(DATASET_PATH, image)    
        # Preprocess input
        input_batch = preprocess_input(image_path, input_size)
        # Predict class
        predicted_class, probability = predict(model, input_batch, class_names)
        # Write to file
        row = [image_path, predicted_class, probability]
        data.append(row)
        cnt += 1
        # print(f"Predicted class: {predicted_class}, Probability: {probability}")
        
    df = pd.DataFrame(data, columns=["image_path", "predicted_class", "probability"])
    df.to_csv("model_predictions_alexnet_1000_samples.csv", index=False)
    print("Done!")