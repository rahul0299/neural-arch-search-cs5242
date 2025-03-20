import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from ChildNetwork import ChildNetwork, train_model, evaluate_model, train_loop # Import the ChildNetwork class and the train_model, evaluate_model, and train_loop functions

# load cifar10 dataset
transform = transforms.Compose( # Compose the transformations
    [transforms.ToTensor(), # Convert the images to tensors
     transforms.Normalize((0.5, ), (0.5, ))] # Normalize the pixel values to be between 0 and 1
)

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform) # Load the training dataset
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform) # Load the test dataset

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) # Create a data loader for the training dataset
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False) # Create a data loader for the test dataset

# Define model architecture
architecture = [
    ["conv", 3, 3, 64], # 3x3 conv, 64 filters
    ["conv", 3, 3, 128], # 3x3 conv, 128 filters
    ["pool", 2, 2], # 2x2 max pooling
    ["dense", 1024], # 1024 neurons
    ["output", 10] # 10 classes
]

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Select the device to run the model on

# Initialize and print model
model = ChildNetwork(architecture).to(device) # Initialize the model
print(model) # Print the model

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # Cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.001 is the learning rate

# Train the model
train_loss, train_time = train_model(model, train_loader, criterion, optimizer, device)
print(f"Training Loss: {train_loss:.4f}, Training Time: {train_time:.2f}s")

# Evaluate the model
accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.4f}")