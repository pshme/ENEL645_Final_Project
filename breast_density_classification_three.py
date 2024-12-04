import torch
import glob
import os
import re
import json
import matplotlib.pylab as plt
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


######################################## Paths and Parameters ########################################

# Train, validation, and test paths for images in /home folder
TRAIN_PATH = r"/home/peter.shmerko/final_project/MW_DATA/TRAIN/output_images"
VAL_PATH = r"/home/peter.shmerko/final_project/MW_DATA/VAL/output_images"
TEST_PATH = r"/home/peter.shmerko/final_project/MW_DATA/TEST/output_images"

batch_size = 12 # Batch size
num_workers = 4 # Number of workers
num_classes = 3 # Number of categories
nepochs = 20 # Number of epochs

PATH_BEST = './garbage_net.pth' # Path to save the best model

######################################## Data Preprocessing ########################################

# ProcessData will take in the path as an input as well as a specific transform, then output a tuple containing the image
# based on the .png input file along with a label from the correspoinding .json file within the folder
class ProcessData(Dataset):
    def __init__(self, root_dir, num_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')] # Load all .png files from root_dir path

        json_class = re.compile(rf".*_{num_classes}\.json$") # Find the correct .json file based on the number of classes
        json_file = [f for f in os.listdir(root_dir) if json_class.match(f)] # Load the .json file from root_dir path
        json_label_file = os.path.join(root_dir, json_file[0]) # Find the path of the .json file

        # Open the .json file
        with open(json_label_file, 'r') as file:
            self.json_data = json.load(file)

        self.type_to_label = {entry["filename"]: entry["label"] for entry in self.json_data} # Scan through the .json file
        self.label_to_int = {"B": 0, "C": 1, "D": 2} # Convert labels to intergers

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        file_name = os.path.basename(image_path) # Extract the file name from the title

        # Extract the first string from the image title before the underscore
        match = re.match(r"([^-_]+)_.*\.png", file_name)
        patient_type = match.group(1)

        label = self.type_to_label.get(patient_type) # Get label from first string in image title
        label = self.label_to_int[label] # Convert label to integer label

        image = Image.open(image_path).convert('RGB') # Opening an image from the path and converting to RGB

        if self.transform:
            image = self.transform(image) # Applying the indicated input transform to the image

        return image, label


# Training transforms
torchvision_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop with resizing
    transforms.GaussianBlur(3),  # Added Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms
torchvision_transform_val = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
    transforms.ToTensor(), # Convert the images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the tensors
])

# Testing transforms
torchvision_transform_test = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
    transforms.ToTensor(), # Convert the images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the tensors
])


# Inputting data into ProcessData class (loading the datasets)
train_dataset = ProcessData(root_dir=TRAIN_PATH, num_classes=num_classes, transform=torchvision_transform_train)
val_dataset = ProcessData(root_dir=VAL_PATH, num_classes=num_classes, transform=torchvision_transform_val)
test_dataset = ProcessData(root_dir=TEST_PATH, num_classes=num_classes, transform=torchvision_transform_test)

# Creating dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


######################################## Image Model ########################################

class ImageModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=True):
        super(ImageModel, self).__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.feature_extractor = models.resnet50(pretrained=True)  # Using pre-trained ResNet50
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1]) # Remove the final fully-connected layer

       # Unfreeze some layers in ResNet50
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_extractor[-4:].parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.feature_extractor(x) # Extract all features and layers from pretrained model
        x = x.view(x.size(0), -1)  # Flatten the feature map while still preserving batch size
        x = self.classifier(x)
        return x


######################################## Training ########################################

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ImageModel(num_classes=3, input_shape=(224, 224)).to(device) # Initializing the model
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001) # Initializing the optimizer
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True) # Initializing the scheduler

# Implementing class weights due to the imbalanced dataset
class_weights = torch.tensor([1.0, 2.0, 2.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Function to train the model for one epoch
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()  # Set the model to train mode
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    return running_loss / len(data_loader), correct / total

# Function to evaluate the model on the validation set
def eval_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    # No need to compute gradients for outputs during validation
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


# Training loop
best_val_accuracy = 0.0
patience = 5
early_stop_count = 0

for epoch in range(nepochs):
    model.train()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Saving the best model if the accuracy has improved or run early stopping
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), PATH_BEST)
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break


print("Training complete!")
print(f"Best model saved at {PATH_BEST}")

# Testing the best model using the test data
model.load_state_dict(torch.load(PATH_BEST))
test_loss, test_accuracy = eval_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")