import optuna
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_dir = '../data/train/'

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Define some constants.
val_split = 0.3

def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    augmentation = trial.suggest_categorical("augmentation",["none","some","more"])
    dropout_rate = trial.suggest_uniform("dropout", 0.0, 0.5)

    # Incorporate trial-specific parameters
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

    model = BinaryClassifier224(dropout_p=dropout_rate)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define loss and dataloaders
    criterion = nn.BCELoss()

    dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    early_stopping = EarlyStopping(patience=100, delta=0.01)

    train_loss_hist = []
    val_loss_hist = []

    # Training loop
    start = time.time()

    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(500):  # Adjust epochs
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

        if epoch % 100 == 0:
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

    np.savetxt("Train_Loss_Hist.csv", np.array(train_loss_hist), delimiter=",")
    np.savetxt("Val_Loss_Hist.csv", np.array(val_loss_hist), delimiter=",")

    print(f"Time taken: {end-start}")
    print(f"Best validation loss: {best_val_loss/len(val_loader):.4f}")

    all_preds = []
    all_labels = []

    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Binary threshold at 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy  # Optuna maximizes this metric

overall_start = time.time()
def progress_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value} and params: {trial.params}")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, callbacks=[progress_callback])

overall_end = time.time()

print(f"Time Taken {overall_end - overall_start}")

print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)

import pickle
import json

# Save the study to a pickle file
with open("study.pkl", "wb") as f:
    pickle.dump(study, f)

all_trials = [
    {
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "state": trial.state.name,
    }
    for trial in study.trials
]

# Save to a JSON file
with open("all_trials.json", "w") as f:
    json.dump(all_trials, f)