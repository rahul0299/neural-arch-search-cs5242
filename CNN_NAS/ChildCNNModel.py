import torch
import torch.nn as nn

import time

import utils

from Fast_Transform import get_fast_transform,normalize_eval_tensor
# from Kornia_Transform import get_kornia_transform,normalize_eval_tensor


# Child model definition and functions
class ChildCNNModel(nn.Module):
    def __init__(self, encoding, input_channels, height, width, output_dim):
        super(ChildCNNModel, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.input_dim = self.input_channels * self.height * self.width
        self.output_dim = output_dim
        self.model = self.build_cnn_model_from_encoding(encoding)

    def forward(self, x):
        return self.model(x)

    def build_cnn_model_from_encoding(self, model_encoding):
        cnn_layers = []
        c, h, w = self.input_channels, self.height, self.width
        convCount = 0

        for layer in model_encoding[0]:
            if len(layer) == 4:
                in_channels, out_channels, kernel_size, padding = layer
                if convCount == 0:
                    in_channels = self.input_channels
                cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
                cnn_layers.append(nn.ReLU())

                h = (h + 2 * padding - kernel_size) + 1
                w = (w + 2 * padding - kernel_size) + 1
                c = out_channels
                convCount += 1

            elif len(layer) == 2:
                kernel_size, stride = layer
                cnn_layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

                h = (h - kernel_size) // stride + 1
                w = (w - kernel_size) // stride + 1

        flattened_dim = c * h * w
        cnn_layers.append(nn.Flatten())

        fc_layers = []
        fc_encoding = model_encoding[1]

        # Replace the first layer's in_features with computed flattened_dim
        fc_layers.append(nn.Linear(flattened_dim, fc_encoding[0][1]))
        fc_layers.append(nn.ReLU())

        for i in range(1, len(fc_encoding) - 1):
            in_features, out_features = fc_encoding[i]
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())

        # Final layer to output_dim
        last_in = fc_encoding[-1][0]
        fc_layers.append(nn.Linear(last_in, self.output_dim))

        return nn.Sequential(*cnn_layers, *fc_layers)

    def train_model(self, data, label, criterion=None, optimizer=None, device=None, epochs=10,dataset_name=None):


        if criterion is None:
            criterion=nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        if device is None:
            device= utils.get_device_available()

        self.model.train()  # Set the model to training mode
        total_loss = 0  # Total loss of the model
        start_time = time.time()  # Start time of the training
        bs = 200
        transform = get_fast_transform(dataset_name)
        # transform = get_kornia_transform(dataset_name)

        self.model.to(device)

        for epoch in range(epochs):
            shuffled_indices = torch.randperm(data.size(0))
            num_batches = 0
            running_loss = 0

            for iter in range(1, len(data), bs):
                num_batches += 1

                # Set dL/dU, dL/dV, dL/dW to be filled with zeros
                optimizer.zero_grad()

                # create a minibatch
                indices = shuffled_indices[iter:iter + bs]
                minibatch_data = data[indices].clone()
                minibatch_label = label[indices]



                # Normalize test set!!!
                # FastAugment
                for j in range(minibatch_data.size(0)):
                    img = minibatch_data[j]
                    if img.max().item() > 1:
                        img = img.float() / 255.0
                    minibatch_data[j] = transform(img.clone().detach())




                # send batch to device
                minibatch_data = minibatch_data.to(device)
                minibatch_label = minibatch_label.to(device)

                # reshape the minibatch
                inputs = minibatch_data.view(-1, self.input_channels, self.height, self.width)

                # tell Pytorch to start tracking all operations that will be done on "inputs"
                inputs.requires_grad_()

                # forward the minibatch through the net
                scores = self.model(inputs)

                # Compute the average of the losses of the data points in the minibatch
                loss = criterion(scores, minibatch_label)
                running_loss += loss.detach().item()

                # backward pass to compute dL/dU, dL/dV and dL/dW
                loss.backward()

                # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
                optimizer.step()

            total_loss = running_loss / num_batches

        elapsed_time = time.time() - start_time
        return total_loss, elapsed_time

    def evaluate_model(self, data, labels, criterion=None, device=None,dataset_name=None):
        self.model.eval()

        if device is None:
            device= utils.get_device_available()

        if criterion is None:
            criterion=nn.CrossEntropyLoss()



        bs = 200
        correct = 0
        total = 0

        running_loss = 0

        with torch.no_grad():
            for i in range(0, data.size(0), bs):
                # Slice the batch manually
                minibatch_data = data[i:i + bs].to(device)
                minibatch_labels = labels[i:i + bs].to(device)


                minibatch_data = normalize_eval_tensor(minibatch_data,dataset_name)


                inputs = minibatch_data.view(-1, self.input_channels, self.height, self.width)

                # Forward pass
                scores = self.model(inputs)
                predicted = torch.argmax(scores, dim=1)

                # Count correct predictions
                total += minibatch_labels.size(0)
                correct += torch.sum(predicted == minibatch_labels).item()

                loss = criterion(scores, minibatch_labels)
                running_loss += loss.item() * minibatch_labels.size(0)

        return correct / total, running_loss / total


    #
    # def loader_train_model(self, dataloader, criterion=None, optimizer=None, device=None, epochs=10):
    #     import time
    #     import torch
    #
    #     if criterion is None:
    #         criterion = nn.CrossEntropyLoss()
    #     if optimizer is None:
    #         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    #     if device is None:
    #         device = utils.get_device_available()
    #
    #     self.model.train()
    #     self.model.to(device)
    #
    #     start_time = time.time()
    #
    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         batch_times = []
    #         num_batches = 0
    #
    #         for inputs, labels in dataloader:
    #             # t0 = time.time()  # batch start time
    #
    #             inputs = inputs.to(device, non_blocking=True)
    #             labels = labels.to(device, non_blocking=True)
    #
    #             optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             loss = criterion(outputs, labels)
    #             running_loss += loss.detach().item()
    #             loss.backward()
    #             optimizer.step()
    #
    #             # batch_time = time.time() - t0
    #             # batch_times.append(batch_time)
    #
    #             num_batches += 1
    #
    #         # avg_batch_time = sum(batch_times) / len(batch_times)
    #         # print(
    #         #     f"[Epoch {epoch + 1}] Loss: {running_loss / num_batches:.4f} | Avg Batch Time: {avg_batch_time:.4f}s | GPU Usage: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
    #
    #     elapsed_time = time.time() - start_time
    #     return running_loss / num_batches, elapsed_time
    #
    # def loader_evaluate_model(self, dataloader, criterion=None, device=None):
    #     if criterion is None:
    #         criterion = nn.CrossEntropyLoss()
    #     if device is None:
    #         device = utils.get_device_available()
    #
    #     self.model.eval()
    #     self.model.to(device)
    #
    #     correct = 0
    #     total = 0
    #     running_loss = 0
    #
    #     with torch.no_grad():
    #         for inputs, labels in dataloader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #
    #             outputs = self.model(inputs)
    #             loss = criterion(outputs, labels)
    #             running_loss += loss.item() * labels.size(0)
    #
    #             preds = outputs.argmax(dim=1)
    #             correct += (preds == labels).sum().item()
    #             total += labels.size(0)
    #
    #     return correct / total, running_loss / total
