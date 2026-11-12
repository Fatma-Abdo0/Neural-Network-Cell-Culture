import torch.nn as nn

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


class FullyConnectedNN(nn.Module):
    """Fully connected network with configurable hidden layers and activation."""

    def __init__(self, n_inputs, n_outputs, hidden_layers, activation):
        super().__init__()
        layers = []
        in_features = n_inputs
        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(ACTIVATIONS[activation]())
            in_features = width
        layers.append(nn.Linear(in_features, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)