import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from Model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

train_data_dir = '../data/train/'
test_data_dir = '../data/test/'

val_split = 0.3
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30), # Randomly rotate images
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust brightness and contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Same normalization as training
])

dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)

val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load Pre-trained ResNet-101
model = models.resnet101(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 1)  # Output layer for binary classification

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=100, delta=0.01)


# Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {val_acc}%")

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

torch.save(model.state_dict(), "resnet101_binary_finetuned.pth")
print("Model saved.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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