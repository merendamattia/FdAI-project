"""
Utilities for analyzing neural network results.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from .utils import get_logger

logger = get_logger()


class ResultsAnalyzer:
    """Analyzes and processes experimental results.

    This class provides functionality for computing metrics and
    comparing results across different experiments for both classification
    and regression tasks.

    Attributes:
        task_type (str): Type of task ('classification' or 'regression')
        results (dict): Dictionary storing results for each experiment
        training_histories (dict): Dictionary storing training history for each experiment
    """

    def __init__(self, task_type='classification'):
        """Initialize the analyzer.

        Args:
            task_type (str, optional): Type of task. Defaults to 'classification'
        """
        self.task_type = task_type
        self.results = {}
        self.training_histories = {}

    def calculate_metrics(self, y_true, y_pred, experiment_name, train_history=None):
        """Calculate performance metrics for an experiment.

        Args:
            y_true (array-like): True target values
            y_pred (array-like): Predicted target values
            experiment_name (str): Name of the experiment
            train_history (dict, optional): Training history. Defaults to None

        Returns:
            dict: Dictionary of calculated metrics
        """
        if self.task_type == 'classification':
            metrics = self._calculate_classification_metrics(y_true, y_pred)
        else:
            metrics = self._calculate_regression_metrics(y_true, y_pred)

        self.results[experiment_name] = metrics

        if train_history is not None:
            self.training_histories[experiment_name] = train_history

        return metrics

    def _calculate_classification_metrics(self, y_true, y_pred):
        """Calculate classification metrics.

        Args:
            y_true (array-like): True class labels
            y_pred (array-like): Predicted class labels

        Returns:
            dict: Dictionary of classification metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics.

        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values

        Returns:
            dict: Dictionary of regression metrics
        """
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }

    def print_results(self):
        """Print results summary and return results DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all results
        """
        if not self.results:
            logger.warning("⚠️ No results available to print.")
            return pd.DataFrame()

        print("\n" + "="*80)
        print("FINAL RESULTS OF EXPERIMENTS")
        print("="*80)

        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)
        print(df_results.to_string())

        # Find best model based on primary metric
        primary_metric = 'f1' if self.task_type == 'classification' else 'r2'
        best_model = df_results[primary_metric].idxmax()
        best_score = df_results.loc[best_model, primary_metric]

        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   {primary_metric.upper()}-Score: {best_score:.4f}")

        return df_results

    def get_results(self):
        """Get results as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all results, or empty DataFrame if no results
        """
        if not self.results:
            logger.warning("⚠️ No results available.")
            return pd.DataFrame()

        df_results = pd.DataFrame(self.results).T
        return df_results.round(4)
