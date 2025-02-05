import torch
import glob
import os
import re
import json
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pylab as plt
from PIL import Image
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import classification_report
import seaborn as sns
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold


######################################## Paths and Parameters ########################################

# Train, validation, and test paths for images in /home folder
TRAIN_PATH = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/TRAIN"
VAL_PATH = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/VAL"
TEST_PATH = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/TEST"

batch_size = 6 # Batch size
num_workers = 4 # Number of workers
num_classes = 4 # Number of categories
nepochs = 20 # Number of epochs

PATH_BEST = './density_net_four.pth' # Path to save the best model

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
        self.label_to_int = {"A": 0, "B": 1, "C": 2, "D": 3} # Convert labels to intergers

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
    transforms.Resize((224, 224)), # Resize the images to 224x224 pixels
    transforms.RandomHorizontalFlip(), # Apply a random horizontal flip to the images
    transforms.RandomVerticalFlip(), # Apply a random vertical flip to the images
    transforms.RandomRotation(15), # Apply a random rotaton of 15 degrees to the images
    transforms.ToTensor(), # Convert the images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the tensors
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

# Combine training and validation datasets into one dataset for k-fold valdation
all_dataset = ConcatDataset([train_dataset, val_dataset])

# Find the labels in the combined dataset
all_labels = []
for idx in range(len(all_dataset)):
    _, label = all_dataset[idx]
    all_labels.append(label)


######################################## Image Model ########################################

class ImageModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=True):
        super(ImageModel, self).__init__()
        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.feature_extractor = resnet50(pretrained=True)  # Using pre-trained ResNet50
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1]) # Remove the final fully-connected layer

       # Unfreeze 4 convolutional layers in ResNet50
        for param in self.feature_extractor[-4:].parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.3) # Apply dropout
        self.classifier = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.feature_extractor(x) # Extract all features and layers from pretrained model
        x = x.view(x.size(0), -1)  # Flatten the feature map while still preserving batch size
        x = self.dropout(x)
        x = self.classifier(x)
        return x


######################################## Training & Validation ########################################

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ImageModel(num_classes=num_classes, input_shape=(224, 224)).to(device) # Initializing the model
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001) # Initializing the optimizer
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True) # Initializing the scheduler

# Implementing class weights due to the imbalanced dataset
class_balance = torch.tensor([0.06, 0.4, 0.31, 0.22]) # Manually importing the category ratios A, B, C, D
class_weights = 1.0 / class_balance # Weights can be calculated as the inverse of the category ratios
class_weights = class_weights / class_weights.sum() # Normalizing the weights

class_weights = class_weights.to(device) # Adding the weights to the model
criterion = nn.CrossEntropyLoss(weight=class_weights) # Implementing a criterion for the model

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

# Function to evaluate the model
def eval_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predictions = []
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

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct / total
    
    return epoch_loss, epoch_accuracy, true_labels, predictions


# Storing k-fold metrics
fold_accuracies = []
fold_losses = []
fold_val_labels = []
fold_val_predictions = []

k_folds = 5 # Performing 5 k-fold runs

# k-fold cross-validation loop
indices = list(range(len(all_dataset)))
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(indices, all_labels)):
    print(f"Fold {fold + 1}/{k_folds}")
    
    # Splitting data into train and validation datasets
    train_subset = torch.utils.data.Subset(all_dataset, train_idx)
    val_subset = torch.utils.data.Subset(all_dataset, val_idx)

    # Initializing the dataloaders for train and validation datasets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Training loop
    best_val_accuracy = 0.0
    patience = 10
    early_stop_count = 0

    for epoch in range(nepochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_predictions = eval_model(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), PATH_BEST) # Saving the best model to the specified path
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load(PATH_BEST))

    # Evaluate validation set
    val_loss, val_accuracy, val_labels, val_predictions = eval_model(model, val_loader, criterion, device)
    fold_losses.append(val_loss)
    fold_accuracies.append(val_accuracy)

    # Save all labels and predictions after each fold
    fold_val_labels.append(val_labels)
    fold_val_predictions.append(val_predictions)

    print(f"Fold {fold + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

# k-fold cross-validation statistics
print("Cross-Validation Complete!")
print(f"Average Validation Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies) * 100:.2f}%")

# Use the model from the k-fold with the best validation accuracy
best_fold_idx = np.argmax(fold_accuracies)
print(f"Best fold is Fold {best_fold_idx + 1} with Validation Accuracy: {fold_accuracies[best_fold_idx] * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(fold_val_labels[best_fold_idx], fold_val_predictions[best_fold_idx], target_names=["A", "B", "C", "D"]))


######################################## Testing ########################################

# Initializing dataloader for test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Testing the best model using the test data
test_loss, test_accuracy, test_labels, test_predictions = eval_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(test_labels, test_predictions, target_names=["A", "B", "C", "D"]))

####################################### Saliency Map (explainability) ########################################

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize original image for correct alignment
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image, image_tensor

def generate_saliency_map(model, image_tensor):
    image_tensor.requires_grad_() # the computational grapgh of the input image is required for computing the derivatives.
    output = model(image_tensor)
    class_idx = torch.argmax(output).item() # the index of the class with the highest classification score
    score = output[0, class_idx]
    
    model.zero_grad() # clear the gradients from the previous computations to avoid accummulating gradients
    score.backward() # back-propagation: compute the gradients of the classification score (prediction probability) with respect to the input image
    
    saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=1) # get the max gradient across the 3 channels (RGB)
    # produces a single-channel saliency map to highlight the most influential pixels
    saliency = saliency.squeeze().cpu().numpy() # remove the batch dimension and send image to the CPU to be plotted as a numpy array
    
    return saliency, class_idx

def save_saliency_map(image_path, model, output_path="saliency_map.png"):
    image, image_tensor = preprocess_image(image_path)
    saliency, class_idx = generate_saliency_map(model, image_tensor)
    idx_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    class_label = idx_to_label.get(class_idx, str(class_idx))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='hot', extent=[0, 224, 0, 224])
    plt.title(f"Saliency Map (Class {class_label})")
    plt.axis("off")
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Provide an image path to save saliency map
A_image_path = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/TEST/LASER1_RCC_sep0p0554937m_1686246092357.png"
B_image_path = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/TEST/WVI53_LCC_sep0p0619619_1729632761088.png" 
C_image_path = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/TEST/STABLE49_RCC_sep0p0709863m_1682033048143.png"
D_image_path = r"/home/anhela.francees/Density_Classification_ML/MW_DATA/TEST/CHEMO16_RCC_sep0p034711900000000007_1723479334097.png"
save_saliency_map(A_image_path, model, "saliency_4class_A.png")
save_saliency_map(B_image_path, model, "saliency_4class_B.png")
save_saliency_map(C_image_path, model, "saliency_4class_C.png")
save_saliency_map(D_image_path, model, "saliency_4class_D.png")
