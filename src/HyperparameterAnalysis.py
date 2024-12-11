import optuna
import pickle

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open('study.pkl', 'rb') as file:
    study = pickle.load(file)

print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)

from Model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_dir = '../data/train/'
test_data_dir = '../data/test/'

val_split = 0.3

batch_size = study.best_params["batch_size"]
lr = study.best_params["lr"]
augmentation = study.best_params["augmentation"]
dropout = study.best_params["dropout"]

if augmentation == "none":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Same normalization as training
    ])

elif augmentation == "some":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30), # Randomly rotate images
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust brightness and contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                  # Resize to target size
        transforms.RandomHorizontalFlip(p=0.5),        
        transforms.RandomVerticalFlip(p=0.5),          
        transforms.RandomRotation(degrees=30),         
        transforms.ColorJitter(brightness=0.3,         
                               contrast=0.3, 
                               saturation=0.3, 
                               hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize values to [-1,1]
    ])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Same normalization as training
])

model = BinaryClassifier224(dropout_p=dropout)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# Define loss and dataloaders
criterion = nn.BCELoss()

dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)

val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

early_stopping = EarlyStopping(patience=100, delta=0.01)

train_loss_hist = []
val_loss_hist = []

# Training loop
start = time.time()

best_val_loss = float('inf')  # Initialize the best validation loss to infinity

for epoch in range(100):  # Adjust epochs
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

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Time Elapsed: {time.time()-start:.4f}")
        print("")

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

print(f"Time taken: {end-start}")
print(f"Best validation loss: {best_val_loss/len(val_loader):.4f}")

all_preds = []
all_labels = []

checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Test Evaluation with Sklearn Metrics
print("\nEvaluating on Test Set...")
model.eval()
true_labels = []
predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        probs = torch.sigmoid(outputs)  # Convert logits to probabilities
        preds = (probs > 0.5).float()  # Convert probabilities to binary predictions

        true_labels.extend(labels.cpu().numpy())
        predictions.extend(preds.cpu().numpy())

true_labels = [label[0] for label in true_labels]  # Convert 2D list to 1D
predictions = [pred[0] for pred in predictions]

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {f1:.2f}")

print("Best hyperparameters:", study.best_params)