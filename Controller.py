# VAIBHAV
#
# TODO: create a vocabulary (For now manual, in future try to be dynamic using a generator)
# NOTE - for simple MLP number range is fine
# TODO: RNN to generate tuples
# [conv, 3x3, 64], [conv, 5x5, 128], [pool, max, 2x2], [dense, 256], [output, softmax, 10]
# start with "[START]" token and finish model with "[END]" token

# CURRENT PLAN
# Generate small model (upto 5 layers) of MLP type.
# For each layer only need to predict the number of neurons in hidden layer (1 hyperparameter)
# Start with bounds 1-10 ie our vocab
# Generate a probability distribution over 1-10 of the next layer containing i neurons [1 <= i <= 10]
# Start generating the sequence by forward pass through RNN
# Use the generated sequence to build and train child network
# Use RL to update RNN weights -- NEXT WEEK (20th Mar+)



### Network Output ->   [["START"], ..... upto 5 numbers, ["END"]]
#### Example1  ->  [["START"], [5], [2], [8], [7], [6], ["END"]]
#### Example2 ->  [["START"], [4], [3], [8], ["END"]]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define vocabulary
VOCAB = {"START": 0, "END": 1, **{str(i): i + 1 for i in range(1, 11)}}  # 1-10 neurons
IDX_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


class RNNController(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32, num_layers=1):
        super(RNNController, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output size = vocab_size

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # Convert token indices to embeddings
        output, hidden = self.rnn(x, hidden)  # Pass through LSTM
        logits = self.fc(output)  # Predict next token
        return logits, hidden


# Function to sample a sequence
def generate_sequence(model, max_layers=5):
    model.eval()
    start_token = torch.tensor([[VOCAB["START"]]])
    hidden = None
    sequence = ["START"]
    input_token = start_token

    for _ in range(max_layers):
        logits, hidden = model(input_token, hidden)
        probs = F.softmax(logits[:, -1, :], dim=-1)  # Convert to probabilities
        sampled_token = torch.multinomial(probs, 1).item()  # Sample from distribution
        sequence.append(IDX_TO_TOKEN[sampled_token])
        if sampled_token == VOCAB["END"]:
            break
        input_token = torch.tensor([[sampled_token]])

    return sequence


# Initialize the model
model = RNNController(VOCAB_SIZE)
print(generate_sequence(model))  # Generate a sample sequence
