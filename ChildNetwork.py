# AGAM
#
# TODO: create the class
# TODO: define a function to build model from encoding
# TODO: define a function to train a model
# TODO: define full training loop
# TODO: improve training function to work with any dataset
# TODO: Gather metrics - latency (time taken to make a prediction), accuracy (on test dataset), training time (time taken to train the model to convergence)

def build_model_from_encoding(encoding):
    """
    Given an encoding (list of layer specifications), build a PyTorch model dynamically.
    Encoding Example:
    [
        ["conv", 3, 3, 64],
        ["conv", 3, 3, 128],
        ["pool", 2, 2],
        ["dense", 1024],
        ["output", 10]
    ]
    """
    layers = [] # list ot store layers dynamically
    input_channels = 3 # assuming image input (CIFAR-10 with 3 RGB channels)
    
    for layer in encoding: # Iterate over the encoding
        layer_type = layer[0] # Get the type of the layer
        if layer_type == "conv": # If layer is a convolutional layer
            _, kernel_size, stride, out_channels = layer # Unpack the layer parameters
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride)) # Apply convolutional layer
            layers.append(nn.ReLU()) # Apply ReLU activation function to the output of the convolutional layer
            input_channels = out_channels # Update the number of input channels for the next layer
        elif layer_type == "pool": # If layer is a pooling layer
            _, kernel_size, stride = layer # Unpack the layer parameters
            layers.append(nn.MaxPool2d(kernel_size, stride)) # Apply max pooling layer
        elif layer_type == "dense": # If layer is a dense / mlp layer
            _, out_features = layer # out_features is the number of neurons in the hidden layer
            layers.append(nn.Flatten()) # Flatten the input to a 1D tensor
            layers.append(nn.Linear(input_channels, out_features)) # Apply linear layer
            layers.append(nn.ReLU()) # Apply ReLU activation function to the output of the linear layer
            input_channels = out_features
        elif layer_type == "output": # If layer is a output layer
            _, out_features = layer # out_features is the number of neurons in the output layer
            layers.append(nn.Linear(input_channels, out_features)) # Linear layer to produce the output
            layers.append(nn.Softmax(dim=1)) # Apply softmax to the output to get a probability distribution over the classes
    
    return nn.Sequential(*layers) # Return the model
       
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader

class ChildNetwork(nn.Module):
    def __init__(self, encoding):
        super(ChildNetwork, self).__init__()
        self.model = build_model_from_encoding(encoding)

    def forward(self, x):
        return self.model(x)
    
def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for a given number of epochs
    """
    model.train() # Set the model to training mode
    total_loss = 0 # Total loss of the model
    start_time = time.time() # Start time of the training

    for inputs, targets, in train_loader: # Iterate over the training data
        inputs, targets = inputs.to(device), targets.to(device) # Move the inputs and targets to the device
        optimizer.zero_grad() # Zero the gradients 
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Calculate the loss
        loss.backward() # Backpropagate the loss
        optimizer.step() # Update the model parameters
        total_loss += loss.item()   # Add the loss to the total loss
    
    elapsed_time = time.time() - start_time 
    # Calculate the time taken to train the model
    return total_loss / len(train_loader), elapsed_time # Return the average loss and the time taken to train the model

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset
    """
    model.eval() # Set the model to evaluation mode
    correct = 0 # Number of correct predictions
    total = 0 # Total number of predictions
    start_time = time.time() # Start time of the evaluation

    with torch.no_grad(): # Disable gradient calculation
        for inputs, targets in test_loader: # Iterate over the test data
            inputs, targets = inputs.to(device), targets.to(device) # Move the inputs and targets to the device
            outputs = model(inputs) # Forward pass
            _, predicted = outputs.max(1) # Get the predicted class
            correct += predicted.eq(targets).sum().item() # Add the number of correct predictions to the total number of predictions
            total += targets.size(0) # Add the number of correct predictions to the total number of predictions
        
    return correct / total # Return the accuracy of the model

def train_loop(architecture_space, train_loader, test_loader, device, num_epochs=10):
    """
    Main training loop using reinforcement learning
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Get the device
    best_accuracy = 0 # Best accuracy of the model
    best_model = None # Best model

    for encoding in architecture_space: # Iterate over the architecture space
        model = ChildNetwork(encoding) # Build the model from the architecture space
        criterion = nn.CrossEntropyLoss() # Define the loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001) # Define the optimizer

        for epoch in range(num_epochs): # Iterate over the number of epochs
            train_loss, train_time = train_model(model, train_loader, criterion, optimizer, device) # Train the model
            accuracy = evaluate_model(model, test_loader, device) # Evaluate the model on the test dataset

            print(f"Encoding: {encoding}, Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {train_time:.2f}s")

            if accuracy > best_accuracy: # If the accuracy is better than the best accuracy
                best_accuracy = accuracy # Update the best accuracy
                best_model = model # Update the best model

    return best_model, best_accuracy # Return the best model and the best accuracy