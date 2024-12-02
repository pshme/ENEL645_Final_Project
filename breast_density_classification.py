import torch
import glob
import matplotlib.pylab as plt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
import re
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


######################################## Paths and Parameters ########################################

# Train, validation, and test paths for images in /work folder
TRAIN_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"

batch_size = 64 # Batch size
num_workers = 4 # Number of workers
num_classes = 4 # Number of classes (labels)
nepochs = 20 # Number of epochs

PATH_BEST = './garbage_net.pth' # Path to save the best model

######################################## Data Preprocessing ########################################

# ProcessData will take in the path as an input as well as a specific transform, then output a tuple containing the image
# based on the .dat input file along with a label from the correspoinding .json file within the folder
class ProcessData(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        
        # Load all .dat files from the directory
        self.dat_file_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.dat')
        ]

        json_file = [f for f in os.listdir(root_dir) if f.endswith('.json')]
        json_label_file = os.path.join(root_dir, json_file[0])
        
        # Load labels from JSON file
        with open(json_label_file, 'r') as file:
            self.labels = json.load(file)

    def __len__(self):
        return len(self.dat_file_paths)

    def __getitem__(self, idx):
        # Get the path and load the .dat file as array
        dat_file_path = self.dat_file_paths[idx]
        image = np.fromfile(dat_file_path, dtype=np.float32).reshape(224,224)
        # Convert heatmap to torch tensor
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        # Get the label from the JSON file
        label_key = dat_file_path.split('/')[-1] 
        label = self.labels[label_key]

        return image, label
        

# Training and validation transforms
torchvision_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
    transforms.RandomHorizontalFlip(), # Apply a random horizontal flip to the images
    transforms.RandomVerticalFlip(), # Apply a random vertical flip to the images
    transforms.RandomRotation(15), # Apply a random rotaton of 15 degrees to the images
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
train_dataset = ProcessData(root_dir=TRAIN_PATH, transform=torchvision_transform)
val_dataset = ProcessData(root_dir=VAL_PATH, transform=torchvision_transform)
test_dataset = ProcessData(root_dir=TEST_PATH, transform=torchvision_transform_test)

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
        
        self.feature_extractor = models.resnet50(pretrained=True) # Loading pretrained ResNet50
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])  # Removing the final fully-connected layer to make ResNet50 more generalizable for this project
        self.classifier = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.feature_extractor(x) # Extract all features and layers from pretrained model
        x = x.view(x.size(0), -1) # Flatten the feature map while still preserving batch size
        x = self.classifier(x)
        #print(f"ImageModel Output Shape {x.shape}") # Debugging for proper shape
        return x


######################################## Training ########################################

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ImageModel(num_classes=4, input_shape=(224,224)).to(device) # Initializing the model
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001) # Initializing the optmizer
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True) # Initializing the scheduler
criterion = nn.CrossEntropyLoss() # Initializing the criterion (loss function)



# Function to train the model for one epoch
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train() # Set the model to train mode
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

# Define a function to evaluate the model on the validation set
def eval_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    # No need to compute gradients for outputs during validation
    with torch.no_grad():
        for images, text_inputs, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Track accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

# Training loop
best_val_accuracy = 0.0
patience = 5
early_stop_count = 0

for epoch in range(nepochs):  # Increased epochs
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)

    # Running the scheduler using validation accuracy
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Saving the best model if the accuracy has improved or run early stopping
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), PATH_BEST)
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