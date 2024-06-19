import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_and_process_data(file_path, batch_size=32):
    # Load dataset
    data = pd.read_csv(file_path)
    data = data.apply(pd.to_numeric, errors='coerce')

    features = data.iloc[:, :-1].values.astype(np.float32)
    labels = data.iloc[:, -1].values.astype(np.int64)  # Assuming labels are integers

    # Reshape features for convolutional layer input
    features = features.reshape(-1, 1, 13, 13)

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader for batch processing
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(169, 256)  # Input size is 13*13=169, output size is 256
        self.fc2 = nn.Linear(256, 128)  # Input size is 256, output size is 128
        self.fc3 = nn.Linear(128, 64)   # Input size is 128, output size is 64
        self.fc4 = nn.Linear(64, 5)     # Input size is 64, output size is 5 (classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)      # Flatten the input
        x = F.relu(self.fc1(x))        # Apply ReLU activation to the first fully connected layer
        x = F.relu(self.fc2(x))        # Apply ReLU activation to the second fully connected layer
        x = F.relu(self.fc3(x))        # Apply ReLU activation to the third fully connected layer
        x = torch.sigmoid(self.fc4(x)) # Apply sigmoid activation to the output layer
        return x

def train_model(model, train_loader, epochs, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    # Load and process data
    train_loader = load_and_process_data('train_audio_features.csv', batch_size=32)
    test_loader = load_and_process_data('test_audio_features.csv', batch_size=32)
    
    # Initialize model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()  # Assuming CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, epochs=300, criterion=criterion, optimizer=optimizer)
    
    # Evaluate on training data
    train_accuracy = evaluate(model, train_loader)
    print('Training Accuracy: {:.2f}%'.format(train_accuracy * 100))
    
    # Evaluate on test data
    test_accuracy = evaluate(model, test_loader)
    print('Testing Accuracy: {:.2f}%'.format(test_accuracy * 100))
