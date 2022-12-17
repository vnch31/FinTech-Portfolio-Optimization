import torch
from torch import nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels,
                 kernel_size, n_dropout, n_timestep, rebalance=True, lower_bound=0, upper_bound=0.3):
        super(TCN, self).__init__()
        n_feature = input_dim
        n_output = output_dim
        self.input_size = n_feature
        # num_chans = [args.hidden_size] * (args.levels - 1) + [args.NumTimeSteps]
        self.tcn = TemporalConvNet(n_feature, num_channels,
                                   kernel_size, dropout=n_dropout)
        self.fc = nn.Linear(num_channels[-1], n_output)
        self.tempmaxpool = nn.MaxPool1d(n_timestep)

        if rebalance:
            self.rebalance = True
            # check bound
            # define lower and upperbound
            if (n_output * lower_bound > 1) or (n_output * upper_bound < 1):
                raise Exception(
                    "Error bounds are not compatible with the number of assets")
            # set bounds
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound


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
        output = self.tcn(x.transpose(1, 2))
        output = self.tempmaxpool(output).squeeze(-1)
        out = self.fc(output)
        out = F.softmax(out, dim=1)
        if self.rebalance:
            out = torch.stack(
                [self.rebalance_weights(weights, self.lower_bound, self.upper_bound) for weights in out])
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        """
        chomp_size: zero padding size
        """
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i

            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)