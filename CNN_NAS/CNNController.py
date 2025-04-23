import torch.nn as nn
import torch

# Define search space
FILTER_CHOICES = [16, 32, 64, 128, 256]
KERNEL_CHOICES = [1, 3, 5]
PADDING_CHOICES = [0, 1, 2]

FILTER_VOCAB = {str(val): idx for idx, val in enumerate(FILTER_CHOICES)}
KERNEL_VOCAB = {str(val): idx for idx, val in enumerate(KERNEL_CHOICES)}
PADDING_VOCAB = {str(val): idx for idx, val in enumerate(PADDING_CHOICES)}

IDX_TO_FILTER = {idx: val for val, idx in FILTER_VOCAB.items()}
IDX_TO_KERNEL = {idx: val for val, idx in KERNEL_VOCAB.items()}
IDX_TO_PADDING = {idx: val for val, idx in PADDING_VOCAB.items()}


import torch
import torch.nn as nn

class CNNController(nn.Module):
    def __init__(self, embedding_dim=8, hidden_dim=32, num_layers=1, name="CNN", meta=None, max_layers=10, type="CNN"):
        super(CNNController, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(FILTER_CHOICES) * len(KERNEL_CHOICES) * len(PADDING_CHOICES),
            embedding_dim=embedding_dim
        )

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.fc_filter = nn.Linear(hidden_dim, len(FILTER_CHOICES))
        self.fc_kernel = nn.Linear(hidden_dim, len(KERNEL_CHOICES))
        self.fc_padding = nn.Linear(hidden_dim, len(PADDING_CHOICES))

        self.meta_layer = nn.Sequential(
            nn.Linear(2, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 2 * hidden_dim, bias=False)
        )

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.name = name
        self.max_layers = max_layers
        self.type = type

        self.meta = torch.tensor([[0.0, 0.0]]) if meta is None else (
            meta if isinstance(meta, torch.Tensor) else torch.tensor([meta], dtype=torch.float)
        )

    def sinusoidal_encoding(self, pos, d_model):
        """
        Generate fixed sinusoidal positional encoding for a given position.
        Returns: [1, 1, d_model] tensor
        """
        pe = torch.zeros(d_model)
        position = torch.tensor(pos, dtype=torch.float32)

        for i in range(0, d_model, 2):
            div_term = torch.tensor(10000.0 ** (i / d_model), dtype=torch.float32)
            pe[i] = torch.sin(position / div_term)
            if i + 1 < d_model:
                pe[i + 1] = torch.cos(position / div_term)

        return pe.view(1, 1, -1)  # [1, 1, d_model]

    def forward(self, input_token, pos_idx, hidden=None):
        # input_token: [1, 1], pos_idx: int or tensor

        if hidden is None:
            meta_encoding = self.meta_layer(self.meta.to(input_token.device))
            h_init = meta_encoding[:, :self.hidden_dim]
            c_init = meta_encoding[:, self.hidden_dim:]
            hidden = (
                h_init.view(self.num_layers, 1, self.hidden_dim),
                c_init.view(self.num_layers, 1, self.hidden_dim)
            )

        x = self.embedding(input_token)  # [1, 1, emb_dim]
        pe = self.sinusoidal_encoding(pos_idx, x.size(-1)).to(x.device)  # [1, 1, emb_dim]
        x = x + pe

        output, hidden = self.rnn(x, hidden)

        filter_logits = self.fc_filter(output)
        kernel_logits = self.fc_kernel(output)
        padding_logits = self.fc_padding(output)

        return filter_logits, kernel_logits, padding_logits, hidden

    def generate_sequence(self, max_layers=1):
        self.eval()
        sequence = []
        log_probs = []

        K = len(KERNEL_CHOICES)
        P = len(PADDING_CHOICES)

        input_token = torch.zeros(1, 1, dtype=torch.long)
        hidden = None

        for layer_idx in range(max_layers):
            f_logits, k_logits, p_logits, hidden = self.forward(input_token, layer_idx, hidden)

            f_dist = torch.distributions.Categorical(logits=f_logits[:, -1, :])
            k_dist = torch.distributions.Categorical(logits=k_logits[:, -1, :])
            p_dist = torch.distributions.Categorical(logits=p_logits[:, -1, :])

            f_token = f_dist.sample()
            k_token = k_dist.sample()
            p_token = p_dist.sample()

            log_prob = f_dist.log_prob(f_token) + k_dist.log_prob(k_token) + p_dist.log_prob(p_token)

            layer = (
                IDX_TO_FILTER[f_token.item()],
                IDX_TO_KERNEL[k_token.item()],
                IDX_TO_PADDING[p_token.item()]
            )
            sequence.append(layer)
            log_probs.append(log_prob)

            # Prepare next input token
            combo_token = f_token * (K * P) + k_token * P + p_token
            input_token = combo_token.view(1, 1)

        return sequence, torch.stack(log_probs)



# Initialize the model
model = CNNController(meta=torch.Tensor([[32, 10]]))
print(model.generate_sequence(max_layers=5))  # Example output
print(model)
del model