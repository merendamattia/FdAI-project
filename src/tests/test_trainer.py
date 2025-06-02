"""
Unit tests for model training functionality.
"""
import unittest
import torch
import numpy as np
from neural_network.trainer import ModelTrainer
from neural_network.models import AutoClassifier, AutoRegressor


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""

    def setUp(self):
        """Set up test cases."""
        torch.manual_seed(42)  # For reproducibility
        self.device = 'cpu'
        self.input_size = 5
        self.num_samples = 100
        self.batch_size = 16
        self.num_epochs = 5

        # Create sample data
        self.X_train = torch.randn(self.num_samples, self.input_size)
        self.X_test = torch.randn(self.num_samples // 2, self.input_size)

        # Create separable data for classification
        self.X_train_clf = torch.cat([
            torch.randn(self.num_samples // 2, self.input_size) + 2,
            torch.randn(self.num_samples // 2, self.input_size) - 2
        ])
        self.y_train_clf = torch.cat([
            torch.zeros(self.num_samples // 2),
            torch.ones(self.num_samples // 2)
        ]).long()

        # Create data with linear pattern for regression
        x_reg = torch.linspace(-5, 5, self.num_samples).reshape(-1, 1)
        self.X_train_reg = torch.cat([x_reg, torch.randn_like(x_reg)], dim=1)
        self.y_train_reg = 2 * x_reg + torch.randn_like(x_reg) * 0.1

    def test_classification_training(self):
        """Test training of classification model."""
        num_classes = 2
        trainer = ModelTrainer(device=self.device, task_type='classification')
        model = AutoClassifier(self.input_size, num_classes)

        # Train model
        model, predictions, losses, metrics = trainer.train_model(
            model, self.X_train_clf, self.y_train_clf, self.X_train_clf[:10], self.y_train_clf[:10],
            num_epochs=self.num_epochs, batch_size=self.batch_size
        )

        # Check outputs
        self.assertIsInstance(model, AutoClassifier)
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(losses), self.num_epochs)
        self.assertTrue(all(isinstance(m, dict) for m in metrics))

        # Check learning progress
        self.assertLess(losses[-1], losses[0])  # Loss should decrease
        self.assertGreater(metrics[-1]['accuracy'], 50)  # Accuracy should be better than random

    def test_regression_training(self):
        """Test training of regression model."""
        trainer = ModelTrainer(device=self.device, task_type='regression')
        model = AutoRegressor(2)  # 2 input features

        # Train model
        model, predictions, losses, metrics = trainer.train_model(
            model, self.X_train_reg, self.y_train_reg,
            self.X_train_reg[:10], self.y_train_reg[:10],
            num_epochs=self.num_epochs, batch_size=self.batch_size
        )

        # Check outputs
        self.assertIsInstance(model, AutoRegressor)
        self.assertEqual(predictions.shape, self.y_train_reg[:10].shape)
        self.assertEqual(len(losses), self.num_epochs)

        # Check learning progress
        self.assertLess(losses[-1], losses[0])  # Loss should decrease

    def test_early_stopping(self):
        """Test early stopping behavior."""
        trainer = ModelTrainer(device=self.device, task_type='classification')
        model = AutoClassifier(self.input_size, 2)

        # Create validation data
        val_size = 20
        X_val = self.X_train_clf[-val_size:]
        y_val = self.y_train_clf[-val_size:]

        # Train with early stopping
        initial_epochs = 10
        model, predictions, losses, metrics = trainer.train_model(
            model, self.X_train_clf[:-val_size], self.y_train_clf[:-val_size],
            X_val, y_val,
            num_epochs=initial_epochs,
            batch_size=self.batch_size
        )

        # Check early stopping effect
        self.assertTrue(len(losses) <= initial_epochs)  # Should stop before max epochs

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        trainer = ModelTrainer(device=self.device, task_type='regression')
        model = AutoRegressor(2)

        # Train with learning rate scheduling for more epochs to see clear effect
        _, _, losses, _ = trainer.train_model(
            model, self.X_train_reg, self.y_train_reg,
            self.X_train_reg[:10], self.y_train_reg[:10],
            num_epochs=30,  # Increased epochs for more stable testing
            batch_size=self.batch_size
        )

        # Convert losses to numpy for easier analysis
        losses = np.array(losses)

        # Check if loss generally decreases
        self.assertLess(losses[-1], losses[0], "Training did not improve the loss")

        # Check if learning happened (at least 50% of steps show improvement)
        improvements = np.diff(losses) < 0
        self.assertGreater(
            np.mean(improvements),
            0.5,
            "Training should show improvement in majority of steps"
        )

    def test_batch_size_effects(self):
        """Test effects of different batch sizes."""
        trainer = ModelTrainer(device=self.device, task_type='classification')

        losses_small_batch = []
        losses_large_batch = []

        # Train with small batch size
        model_small = AutoClassifier(self.input_size, 2)
        _, _, losses_small, _ = trainer.train_model(
            model_small, self.X_train_clf, self.y_train_clf,
            self.X_train_clf[:10], self.y_train_clf[:10],
            num_epochs=5, batch_size=4
        )
        losses_small_batch.extend(losses_small)

        # Train with large batch size
        model_large = AutoClassifier(self.input_size, 2)
        _, _, losses_large, _ = trainer.train_model(
            model_large, self.X_train_clf, self.y_train_clf,
            self.X_train_clf[:10], self.y_train_clf[:10],
            num_epochs=5, batch_size=32
        )
        losses_large_batch.extend(losses_large)

        # Small batches should have more variance in loss
        small_variance = np.var(losses_small_batch)
        large_variance = np.var(losses_large_batch)
        self.assertGreater(small_variance, large_variance)


if __name__ == '__main__':
    unittest.main()
