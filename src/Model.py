import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier224(nn.Module):
    def __init__(self,dropout_p=0.5):
        super(BinaryClassifier224, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_p)  # Drop 50% of neurons

        # Fully connected layers
        self.fc1 = nn.Linear(28 * 28 * 64, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 1)  # Binary classification (1 output neuron)

    def forward(self, x):
        # Convolutional + Batch Normalization + ReLU + Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 28*28*64)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset patience counter
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False

def save_checkpoint(state, filename="best_model.pth"):
    """Save the model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")