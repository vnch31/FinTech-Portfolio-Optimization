import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, rebalance=True, lower_bound=0, upper_bound=0.3):
        super(RNN, self).__init__()

        if rebalance:
            self.rebalance = True
            # check bound
            # define lower and upperbound
            if (input_dim * lower_bound > 1) or (input_dim * upper_bound < 1):
                raise Exception(
                    "Error bounds are not compatible with the number of assets")
            # set bounds
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        # Define number of layers and node in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN Stack
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layer_dim,
            dropout=dropout_prob,
            batch_first=True
        )

        # Fully connected layer to output 1 dim
        self.fc = nn.Linear(hidden_dim, output_dim)

        # softmax
        self.sm = nn.Softmax(dim=1)

    def rebalance_weights(self, weight, lb, ub):
        old = weight
        weight_clamped = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - weight_clamped).sum().item()
            nominees = weight_clamped[torch.where(weight_clamped != ub)[0]]
            gift = leftover * (nominees / nominees.sum())
            weight_clamped[torch.where(weight_clamped != ub)[0]] += gift
            old = weight_clamped
            if len(torch.where(weight_clamped > ub)[0]) == 0:
                break
            else:
                weight_clamped = torch.clamp(old, lb, ub)
        return weight_clamped

    def forward(self, x):
        # Forward propagation
        out, _ = self.rnn(x)

        # Reshaping outputs in shape (batch, seq, hidden)
        out = out[:, -1, :]

        # convert final state to output (batch, output_dim)
        out = self.fc(out)

        # Softmax to apply constraints to the weights
        out = self.sm(out)

        # rebalance weights
        if self.rebalance:
            out = torch.stack(
                [self.rebalance_weights(weights, self.lower_bound, self.upper_bound) for weights in out])

        return out
