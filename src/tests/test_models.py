"""
Unit tests for neural network models.
"""
import unittest
import torch
import numpy as np
from neural_network.models import AutoClassifier, AutoRegressor
from neural_network.models.base import BaseNeuralNetwork


class TestBaseNeuralNetwork(unittest.TestCase):
    """Test cases for BaseNeuralNetwork."""

    def setUp(self):
        """Set up test cases."""
        self.input_size = 10
        self.model = BaseNeuralNetwork(self.input_size)

    def test_initialization(self):
        """Test base model initialization."""
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(len(self.model.hidden_sizes), 3)
        self.assertEqual(len(self.model.dropout_rates), 3)

    def test_build_layers(self):
        """Test layer construction."""
        # Test number of layers (3 blocks of Linear+BatchNorm+ReLU+Dropout)
        self.assertEqual(len(list(self.model.hidden_layers)), 12)

        # Test layer types
        layers = list(self.model.hidden_layers)
        self.assertIsInstance(layers[0], torch.nn.Linear)
        self.assertIsInstance(layers[1], torch.nn.BatchNorm1d)
        self.assertIsInstance(layers[2], torch.nn.ReLU)
        self.assertIsInstance(layers[3], torch.nn.Dropout)

    def test_weight_initialization(self):
        """Test Xavier initialization."""
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Linear):
                # Check if weights follow Xavier distribution
                std = np.sqrt(2.0 / (layer.in_features + layer.out_features))
                weight_std = layer.weight.std().item()
                self.assertAlmostEqual(weight_std, std, places=1)

                # Check if biases are initialized to zero
                self.assertTrue(torch.allclose(layer.bias, torch.zeros_like(layer.bias)))

    def test_architecture_info(self):
        """Test get_architecture_info method."""
        info = self.model.get_architecture_info()
        self.assertIn('input_size', info)
        self.assertIn('hidden_sizes', info)
        self.assertIn('dropout_rates', info)
        self.assertIn('task_type', info)


class TestAutoClassifier(unittest.TestCase):
    """Test cases for AutoClassifier model."""

    def setUp(self):
        """Set up test cases."""
        self.input_size = 10
        self.num_classes = 3
        self.batch_size = 4
        self.model = AutoClassifier(self.input_size, self.num_classes)

    def test_model_initialization(self):
        """Test model initialization and architecture."""
        self.assertIsInstance(self.model.hidden_layers[0], torch.nn.Linear)
        self.assertEqual(self.model.hidden_layers[0].in_features, self.input_size)
        self.assertEqual(self.model.output_layer.out_features, self.num_classes)

    def test_forward_pass(self):
        """Test forward pass through the model."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_output_range(self):
        """Test if output logits are in reasonable range."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.model(x)
        self.assertTrue(torch.all(output > -100) and torch.all(output < 100))

    def test_gradient_flow(self):
        """Test if gradients flow through the model."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # Check if all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_model_training_mode(self):
        """Test model training/eval mode behavior."""
        x = torch.randn(self.batch_size, self.input_size)

        # Test training mode
        self.model.train()
        out1 = self.model(x)
        out2 = self.model(x)
        self.assertFalse(torch.allclose(out1, out2))  # Dropout should make outputs different

        # Test eval mode
        self.model.eval()
        out1 = self.model(x)
        out2 = self.model(x)
        self.assertTrue(torch.allclose(out1, out2))  # Should be deterministic


class TestAutoRegressor(unittest.TestCase):
    """Test cases for AutoRegressor model."""

    def setUp(self):
        """Set up test cases."""
        self.input_size = 10
        self.batch_size = 4
        self.model = AutoRegressor(self.input_size)

    def test_model_initialization(self):
        """Test model initialization and architecture."""
        self.assertIsInstance(self.model.hidden_layers[0], torch.nn.Linear)
        self.assertEqual(self.model.hidden_layers[0].in_features, self.input_size)
        self.assertEqual(self.model.output_layer.out_features, 1)

    def test_forward_pass(self):
        """Test forward pass through the model."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_gradient_flow(self):
        """Test if gradients flow through the model."""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients for each layer
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

    def test_batch_normalization(self):
        """Test batch normalization behavior."""
        x = torch.randn(100, self.input_size)  # Larger batch for stable statistics

        # Training mode
        self.model.train()
        out_train = self.model(x)

        # Get statistics from first batch norm layer
        bn_layer = next(m for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm1d))
        mean_train = bn_layer.running_mean.clone()
        var_train = bn_layer.running_var.clone()

        # Eval mode should use running statistics
        self.model.eval()
        out_eval = self.model(x)
        self.assertFalse(torch.allclose(out_train, out_eval))

        # Running stats should be frozen in eval mode
        self.assertTrue(torch.allclose(mean_train, bn_layer.running_mean))
        self.assertTrue(torch.allclose(var_train, bn_layer.running_var))

    def test_model_capacity(self):
        """Test if model can fit a simple regression pattern."""
        # Create a simple linear pattern with noise, repeating to match input size
        base_x = torch.linspace(-5, 5, 100).reshape(-1, 1)
        x = base_x.repeat(1, self.input_size)  # Repeat the pattern across all input dimensions
        y = 2 * base_x + torch.randn_like(base_x) * 0.1

        # Train for a few steps
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.MSELoss()

        initial_loss = None
        final_loss = None

        for _ in range(50):
            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # Model should improve
        self.assertLess(final_loss, initial_loss)


if __name__ == '__main__':
    unittest.main()
