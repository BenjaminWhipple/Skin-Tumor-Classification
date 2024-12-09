import torch
import time

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from Model import *


# Path to your data folders
train_data_dir = '../data/train/'
test_data_dir = '../data/test/'
# Transformations for the images, we try to be extremely aggressive in order to improve generalizability.
#"""
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(30),     # Randomly rotate images
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust brightness and contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Same normalization as training
])

# Load the dataset
dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)

# Split into training and validation sets
val_split = 0.3
val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)


# Data loaders
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=2)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# Print class names
print(f"Class names: {dataset.classes}")


# Instantiate the model
model = BinaryClassifier224()

# Print model summary
print(model)

import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

early_stopping = EarlyStopping(patience=100, delta=0.01)

best_val_loss = float('inf')  # Initialize the best validation loss to infinity

# Training loop
train_loss_hist = []
val_loss_hist = []

start = time.time()
for epoch in range(1000):  # Adjust epochs
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Adjust labels shape
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss
        }
        save_checkpoint(checkpoint, filename="best_model.pth")

    train_loss_hist.append(train_loss/len(train_loader))
    val_loss_hist.append(val_loss/len(val_loader))

end = time.time()

np.savetxt("Train_Loss_Hist.csv", np.array(train_loss_hist), delimiter=",")
np.savetxt("Val_Loss_Hist.csv", np.array(val_loss_hist), delimiter=",")

print(f"Time taken: {end-start}")
print(f"Best validation loss: {best_val_loss/len(val_loader):.4f}")

### Evaluate trained model.

# Load the model checkpoint
model = BinaryClassifier224()  # Replace with your model class
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Store all predictions and true labels
all_preds = []
all_labels = []

# Disable gradient computation for evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        # Move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Get model predictions
        outputs = model(inputs)
        preds = (outputs > 0.5).float()  # Binary threshold at 0.5

        # Collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
