"""
Neural network models for classification and regression tasks.
"""
import torch.nn as nn
from .base import BaseNeuralNetwork


class AutoClassifier(BaseNeuralNetwork):
    """Automatic neural network for classification tasks.

    This class extends the base neural network for classification problems with
    automatic architecture determination and output layer configuration.

    Attributes:
        num_classes (int): Number of output classes
    """

    def __init__(self, input_size, num_classes):
        """Initialize the classification network.

        Args:
            input_size (int): Number of input features
            num_classes (int): Number of output classes
        """
        super(AutoClassifier, self).__init__(input_size, task_type='classification')
        self.num_classes = num_classes
        self.output_layer = nn.Linear(self.hidden_sizes[-1], num_classes)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Class logits
        """
        x = super().forward(x)
        return self.output_layer(x)


class AutoRegressor(BaseNeuralNetwork):
    """Automatic neural network for regression tasks.

    This class extends the base neural network for regression problems with
    automatic architecture determination and output layer configuration.
    """

    def __init__(self, input_size):
        """Initialize the regression network.

        Args:
            input_size (int): Number of input features
        """
        super(AutoRegressor, self).__init__(input_size, task_type='regression')
        self.output_layer = nn.Linear(self.hidden_sizes[-1], 1)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Predicted values
        """
        x = super().forward(x)
        return self.output_layer(x)
