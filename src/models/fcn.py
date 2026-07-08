import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    Fully Connected Neural Network (MLP) for PINNs.
    """
    def __init__(self, layers):
        """
        :param layers: List of integers, e.g., [2, 50, 50, 50, 1]
                       Input Dim -> Hidden -> ... -> Output Dim
        """
        super(FCN, self).__init__()

        self.layers = layers
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *inputs):
        # Concatenate all inputs to form (N, InIterface)
        a = torch.cat(inputs, dim=1)

        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)

        # Last layer: no activation (usually) or linear
        a = self.linears[-1](a)
        return a

    def extract_features(self, *inputs, layer='penultimate'):
        """Extract hidden features for transfer regularization baselines."""
        a = torch.cat(inputs, dim=1)
        hidden_features = []

        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
            hidden_features.append(a)

        if not hidden_features:
            return a

        if layer == 'penultimate':
            return hidden_features[-1]

        if isinstance(layer, int):
            return hidden_features[layer]

        raise ValueError(f"Unsupported feature layer: {layer}")
