import torch
from torch import nn

class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, dim_ff, num_layers, output_dim, dropout_prob, rebalance=True, lower_bound=0, upper_bound=0.3):
        super(TransformerEncoder, self).__init__()

        # Define number of layers and node in each layer
        self.d_model = input_dim
        self.n_head = hidden_dim

        if rebalance:
            self.rebalance = True
            # set bounds
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        # Define number of layers and node in each layer
        self.hidden_dim = hidden_dim


        # Encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dropout=dropout_prob,
            dim_feedforward=dim_ff,
            batch_first=True
        )
        
        # transformer encoder stack
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Fully connected layer to output 1 dim
        self.fc = nn.Linear(input_dim, output_dim)
        
        # softmax layer
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
        out = self.transformer_encoder(x)
        
        # Reshaping outputs in shape (batch, seq, hidden)
        out = out[:, -1, :]
        
        # Pass through FC Layer
        out = self.fc(out)
    
        # apply constraints to weights
        out = self.sm(out)

        if self.rebalance:
            out = torch.stack(
                [self.rebalance_weights(weights, self.lower_bound, self.upper_bound) for weights in out])

        return out