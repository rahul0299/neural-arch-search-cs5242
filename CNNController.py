import torch
import torch.nn as nn
import torch.nn.functional as F

# Define search space
FILTER_CHOICES = [16, 32, 64, 128]
KERNEL_CHOICES = [1, 3, 5]
PADDING_CHOICES = [0, 1, 2]

FILTER_VOCAB = {str(val): idx for idx, val in enumerate(FILTER_CHOICES)}
KERNEL_VOCAB = {str(val): idx for idx, val in enumerate(KERNEL_CHOICES)}
PADDING_VOCAB = {str(val): idx for idx, val in enumerate(PADDING_CHOICES)}

IDX_TO_FILTER = {idx: val for val, idx in FILTER_VOCAB.items()}
IDX_TO_KERNEL = {idx: val for val, idx in KERNEL_VOCAB.items()}
IDX_TO_PADDING = {idx: val for val, idx in PADDING_VOCAB.items()}

class CNNController(nn.Module):
    def __init__(self, embedding_dim=8, hidden_dim=32, num_layers=1):
        super(CNNController, self).__init__()
        self.embedding = nn.Embedding(1, embedding_dim)  # Dummy embedding for input
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_filter = nn.Linear(hidden_dim, len(FILTER_CHOICES))
        self.fc_kernel = nn.Linear(hidden_dim, len(KERNEL_CHOICES))
        self.fc_padding = nn.Linear(hidden_dim, len(PADDING_CHOICES))

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        filter_logits = self.fc_filter(output)
        kernel_logits = self.fc_kernel(output)
        padding_logits = self.fc_padding(output)
        return filter_logits, kernel_logits, padding_logits, hidden

    def generate_sequence(self, max_layers=1):
        self.eval()
        input_token = torch.zeros(1, 1, dtype=torch.long)  # Dummy start token
        hidden = None
        sequence = []
        log_probs = []

        for _ in range(max_layers):
            f_logits, k_logits, p_logits, hidden = self.forward(input_token, hidden)

            f_dist = torch.distributions.Categorical(logits=f_logits[:, -1, :])
            k_dist = torch.distributions.Categorical(logits=k_logits[:, -1, :])
            p_dist = torch.distributions.Categorical(logits=p_logits[:, -1, :])

            f_token = f_dist.sample()
            k_token = k_dist.sample()
            p_token = p_dist.sample()

            log_prob = f_dist.log_prob(f_token) + k_dist.log_prob(k_token) + p_dist.log_prob(p_token)

            layer = (IDX_TO_FILTER[f_token.item()], IDX_TO_KERNEL[k_token.item()], IDX_TO_PADDING[p_token.item()])
            sequence.append(layer)
            log_probs.append(log_prob)

        return sequence, torch.stack(log_probs)

# Initialize the model
model = CNNController()
x = model.generate_sequence()
print(x)
test_sequence, _ = x
print(test_sequence)
