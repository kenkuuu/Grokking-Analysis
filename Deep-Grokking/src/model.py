import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with ReLU activations.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        output_dim: int,
        alpha: float = 1.0
    ):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden layers.
            depth (int): The number of hidden layers.
            output_dim (int): Dimensionality of the output.
            alpha (float): A scaling factor for the initial weights.
        """
        super().__init__()
        
        layers = []
        # Create `depth` number of hidden layers
        for i in range(depth - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights(alpha)

    def _init_weights(self, alpha: float):
        """
        Initializes the weights of the linear layers.

        Weights are initialized using Kaiming normal initialization and then
        scaled by a factor `alpha`. Biases are initialized to zero.

        Args:
            alpha (float): The scaling factor for the weights.
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # Scale weights by alpha
                m.weight.data.mul_(alpha)
                # Initialize biases to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Performs the forward pass of the model."""
        return self.net(x)