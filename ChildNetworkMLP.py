input_dim = 0
output_dim = 0

def set_data_params(x,y):
    global input_dim, output_dim
    input_dim = x
    output_dim = y


def build_model_from_encoding(encoding):
    """
    Given an encoding (list of layer specifications), build a PyTorch model dynamically.
    Encoding Example:

[["START"], [4], [3], [8], ["END"]]
    """
    layers = []  # list ot store layers dynamically
    sticky_dim = input_dim
    print(sticky_dim)
    # TODO: If START found elsewhere give negative feedback and dont create model
    for layer in encoding:  # Iterate over the encoding
        layer_param = layer  # Get the type of the layer
        if layer_param.isnumeric():
            layer_param = int(layer_param)
            layers.append(nn.Linear(sticky_dim, layer_param))
            layers.append(nn.ReLU())
            sticky_dim = layer_param
        if layer_param == "END":
            layers.append(nn.Linear(sticky_dim, output_dim))
            layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)  # Return the model


import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader


class ChildModel(nn.Module):
    def __init__(self, encoding):
        super(ChildModel, self).__init__()
        self.model = build_model_from_encoding(encoding)

    def forward(self, x):
        return self.model(x)

def train_model_v2(model, data,label, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0  # Total loss of the model
    start_time = time.time()  # Start time of the training
    bs = 200

    for iter in range(1, len(data),bs):
        # Set dL/dU, dL/dV, dL/dW to be filled with zeros
        optimizer.zero_grad()

        # create a minibatch
        indices = torch.LongTensor(bs).random_(0, len(data))
        minibatch_data = data[indices]
        minibatch_label = label[indices]

        # reshape the minibatch
        inputs = minibatch_data.view(bs, input_dim)

        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net
        scores = model(inputs)

        # Compute the average of the losses of the data points in the minibatch
        loss = criterion(scores, minibatch_label)

        # backward pass to compute dL/dU, dL/dV and dL/dW
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()

        total_loss += loss.detach().item()


    elapsed_time = time.time() - start_time
    return total_loss / len(data), elapsed_time



def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for a given number of epochs
    """
    model.train()  # Set the model to training mode
    total_loss = 0  # Total loss of the model
    start_time = time.time()  # Start time of the training

    for inputs, targets, in train_loader:  # Iterate over the training data
        inputs, targets = inputs.to(device), targets.to(device)  # Move the inputs and targets to the device
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters
        total_loss += loss.item()  # Add the loss to the total loss

    elapsed_time = time.time() - start_time
    # Calculate the time taken to train the model
    return total_loss / len(train_loader), elapsed_time  # Return the average loss and the time taken to train the model


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Number of correct predictions
    total = 0  # Total number of predictions
    start_time = time.time()  # Start time of the evaluation

    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in test_loader:  # Iterate over the test data
            inputs, targets = inputs.to(device), targets.to(device)  # Move the inputs and targets to the device
            outputs = model(inputs)  # Forward pass
            _, predicted = outputs.max(1)  # Get the predicted class
            correct += predicted.eq(
                targets).sum().item()  # Add the number of correct predictions to the total number of predictions
            total += targets.size(0)  # Add the number of correct predictions to the total number of predictions

    return correct / total  # Return the accuracy of the model


def train_loop(architecture_space, train_loader, test_loader, device, num_epochs=10):
    """
    Main training loop using reinforcement learning
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the device
    best_accuracy = 0  # Best accuracy of the model
    best_model = None  # Best model

    for encoding in architecture_space:  # Iterate over the architecture space
        model = ChildModel(encoding)  # Build the model from the architecture space
        criterion = nn.CrossEntropyLoss()  # Define the loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer

        for epoch in range(num_epochs):  # Iterate over the number of epochs
            train_loss, train_time = train_model(model, train_loader, criterion, optimizer, device)  # Train the model
            accuracy = evaluate_model(model, test_loader, device)  # Evaluate the model on the test dataset

            print(
                f"Encoding: {encoding}, Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {train_time:.2f}s")

            if accuracy > best_accuracy:  # If the accuracy is better than the best accuracy
                best_accuracy = accuracy  # Update the best accuracy
                best_model = model  # Update the best model

    return best_model, best_accuracy  # Return the best model and the best accuracy
