"""
Model training utilities for neural networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from .utils import get_logger

logger = get_logger()


class ModelTrainer:
    """Handles neural network training and evaluation.

    This class provides functionality for training both classification and regression
    models with configurable hyperparameters and training settings.

    Attributes:
        device (str): Device to run training on ('cpu' or 'cuda')
        task_type (str): Type of task ('classification' or 'regression')
    """

    def __init__(self, device='cpu', task_type='classification'):
        """Initialize the trainer.

        Args:
            device (str, optional): Device to use. Defaults to 'cpu'
            task_type (str, optional): Type of task. Defaults to 'classification'
        """
        self.device = device
        self.task_type = task_type

    def train_model(self, model, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32):
        """Train a neural network model.

        Args:
            model (nn.Module): Neural network model to train
            X_train (torch.Tensor): Training features
            y_train (torch.Tensor): Training targets
            X_test (torch.Tensor): Test features
            y_test (torch.Tensor): Test targets
            num_epochs (int, optional): Number of training epochs. Defaults to 100
            batch_size (int, optional): Batch size. Defaults to 32

        Returns:
            tuple: (model, predictions, train_losses, train_metrics)
        """
        # Define the loss function based on the task type (classification or regression)
        criterion = (nn.CrossEntropyLoss() if self.task_type == 'classification'
                    else nn.MSELoss())

        # Initialize the Adam optimizer with learning rate and weight decay
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Set up a learning rate scheduler that decays the LR every 30 epochs by a factor of 0.1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Wrap training data into a TensorDataset
        train_dataset = TensorDataset(X_train, y_train)
        # Create a DataLoader to iterate over training data in batches
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training history
        train_losses = []
        train_metrics = []

        # Set the model to training mode
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_metrics = self._init_epoch_metrics()

            # Iterate over batches in the training data
            for batch_X, batch_y in train_loader:
                # Move data to the configured device (CPU or GPU)
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = model(batch_X)
                # Compute the loss between predictions and true labels
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                self._update_epoch_metrics(epoch_metrics, outputs, batch_y)

            # Step the learning rate scheduler
            scheduler.step()

            # Calculate average metrics
            avg_loss = epoch_loss / len(train_loader)
            avg_metrics = self._calculate_avg_metrics(epoch_metrics, len(train_loader))

            train_losses.append(avg_loss)
            train_metrics.append(avg_metrics)

            if (epoch + 1) % 5 == 0:
                self._log_training_progress(epoch + 1, num_epochs, avg_loss, avg_metrics)

        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(self.device))
            predictions = (torch.max(test_outputs, 1)[1] if self.task_type == 'classification'
                         else test_outputs).cpu()

        return model, predictions, train_losses, train_metrics

    def _init_epoch_metrics(self):
        """Initialize metrics for an epoch based on task type.

        Returns:
            dict: Initialized metrics dictionary
        """
        if self.task_type == 'classification':
            return {'correct': 0, 'total': 0}
        return {}

    def _update_epoch_metrics(self, metrics, outputs, targets):
        """Update metrics for the current batch.

        Args:
            metrics (dict): Current metrics dictionary
            outputs (torch.Tensor): Model outputs
            targets (torch.Tensor): True targets
        """
        if self.task_type == 'classification':
            _, predicted = torch.max(outputs.data, 1)
            metrics['total'] += targets.size(0)
            metrics['correct'] += (predicted == targets).sum().item()

    def _calculate_avg_metrics(self, metrics, num_batches):
        """Calculate average metrics for the epoch.

        Args:
            metrics (dict): Current metrics dictionary
            num_batches (int): Number of batches in epoch

        Returns:
            dict: Dictionary of averaged metrics
        """
        if self.task_type == 'classification':
            return {'accuracy': 100 * metrics['correct'] / metrics['total']}
        return {}

    def _log_training_progress(self, epoch, num_epochs, loss, metrics):
        """Log training progress.

        Args:
            epoch (int): Current epoch number
            num_epochs (int): Total number of epochs
            loss (float): Current loss value
            metrics (dict): Current metrics dictionary
        """
        metric_str = ''
        if self.task_type == 'classification':
            metric_str = f", Accuracy: {metrics['accuracy']:.2f}%"

        logger.info(f'Epoch [{epoch}/{num_epochs}], Loss: {loss:.4f}{metric_str}')
