import torch.nn as nn
import torch

# Define search space
FILTER_CHOICES = [32, 64, 128, 256]
KERNEL_CHOICES = [1, 3, 5]
PADDING_CHOICES = [0, 1, 2]

FILTER_VOCAB = {str(val): idx for idx, val in enumerate(FILTER_CHOICES)}
KERNEL_VOCAB = {str(val): idx for idx, val in enumerate(KERNEL_CHOICES)}
PADDING_VOCAB = {str(val): idx for idx, val in enumerate(PADDING_CHOICES)}

IDX_TO_FILTER = {idx: val for val, idx in FILTER_VOCAB.items()}
IDX_TO_KERNEL = {idx: val for val, idx in KERNEL_VOCAB.items()}
IDX_TO_PADDING = {idx: val for val, idx in PADDING_VOCAB.items()}


class CNNController(nn.Module):
    def __init__(self, embedding_dim=8, hidden_dim=32, num_layers=1, name="CNN", meta=None):
        super(CNNController, self).__init__()

        self.meta_layer = nn.Sequential(
            nn.Linear(2, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 2 * hidden_dim, bias=False)
        )

        self.vocab_size = len(FILTER_CHOICES) * len(KERNEL_CHOICES) * len(PADDING_CHOICES)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.fc_filter = nn.Linear(hidden_dim, len(FILTER_CHOICES))
        self.fc_kernel = nn.Linear(hidden_dim, len(KERNEL_CHOICES))
        self.fc_padding = nn.Linear(hidden_dim, len(PADDING_CHOICES))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.name = name

        if meta is None:
            # default to [0, 0] if no metadata provided
            self.meta = torch.tensor([[0.0, 0.0]])
        else:
            self.meta = meta if isinstance(meta, torch.Tensor) else torch.tensor([meta], dtype=torch.float)


    def forward(self, x, hidden=None):
        if hidden is None:
            if self.meta is not None:
                meta_encoding = self.meta_layer(self.meta.to(x.device))
                h_init = meta_encoding[:, :self.hidden_dim]
                c_init = meta_encoding[:, self.hidden_dim:]
                hidden = (
                    h_init.view(1, 1, self.hidden_dim),
                    c_init.view(1, 1, self.hidden_dim)
                )
            else:
                # No metadata → use zeros (default behavior)
                h_init = torch.zeros(1, 1, self.hidden_dim, device=x.device)
                c_init = torch.zeros(1, 1, self.hidden_dim, device=x.device)
                hidden = (h_init, c_init)

        x = self.embedding(x)
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

        # with torch.no_grad():
        input_token = torch.zeros(1, 1, dtype=torch.long)
        hidden = None

        for _ in range(max_layers):
            input_token = input_token.detach()
            f_logits, k_logits, p_logits, hidden = self.forward(input_token, hidden)

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

            combo_token = f_token * (K * P) + k_token * P + p_token
            input_token = combo_token.view(1, 1)

        return sequence, torch.stack(log_probs)




# Initialize the model
model = CNNController(meta=torch.Tensor([[32, 10]]))
print(model.generate_sequence(max_layers=5))  # Example output
print(model)
del model