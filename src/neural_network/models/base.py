"""
Base neural network model class that provides common functionality.
"""
import torch
import torch.nn as nn


class BaseNeuralNetwork(nn.Module):
    """Base neural network architecture with automatic layer sizing.

    This class provides the foundation for both classification and regression models
    with automatic architecture determination based on input size.

    Attributes:
        task_type (str): Type of task ('classification' or 'regression')
        input_size (int): Number of input features
        hidden_sizes (list): Sizes of hidden layers
        dropout_rates (list): Dropout rates for each layer
    """

    def __init__(self, input_size, task_type='classification'):
        """Initialize the base neural network.

        Args:
            input_size (int): Number of input features
            task_type (str, optional): Type of task. Defaults to 'classification'
        """
        super(BaseNeuralNetwork, self).__init__()
        self.task_type = task_type
        self.input_size = input_size

        # Compute hidden layer sizes
        self.hidden_sizes = [
            max(64, input_size * 2),
            max(32, input_size),
            max(16, input_size // 2)
        ]

        # Dropout rates decrease with depth
        self.dropout_rates = [0.3, 0.2, 0.1]

        # Build network architecture
        self._build_layers()
        self._initialize_weights()

    def _build_layers(self):
        """Construct the network layers with batch normalization and dropout."""
        layers = []
        current_size = self.input_size

        # Build hidden layers
        for hidden_size, dropout_rate in zip(self.hidden_sizes, self.dropout_rates):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size

        self.hidden_layers = nn.ModuleList(layers)

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def get_architecture_info(self):
        """Get the model's architecture information.

        Returns:
            dict: Dictionary containing model architecture details
        """
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rates': self.dropout_rates,
            'task_type': self.task_type
        }

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.hidden_layers:
            x = layer(x)
        return x
