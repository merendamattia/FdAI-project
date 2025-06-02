"""
Unit tests for results analysis functionality.
"""
import unittest
import os
import numpy as np
from neural_network.analyzer import ResultsAnalyzer


class TestResultsAnalyzer(unittest.TestCase):
    """Test cases for ResultsAnalyzer class."""

    def setUp(self):
        """Set up test cases."""
        # Create sample data
        self.y_true = np.array([0, 1, 0, 1, 1])
        self.y_pred = np.array([0, 1, 1, 1, 0])
        self.experiment_name = 'test_experiment'

        # Sample training history
        self.train_history = {
            'losses': [0.5, 0.3, 0.2],
            'accuracy': [0.7, 0.8, 0.85]
        }

    def test_classification_metrics(self):
        """Test calculation of classification metrics."""
        analyzer = ResultsAnalyzer(task_type='classification')
        metrics = analyzer.calculate_metrics(
            self.y_true, self.y_pred,
            self.experiment_name,
            self.train_history
        )

        # Check if all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)

    def test_regression_metrics(self):
        """Test calculation of regression metrics."""
        # Create regression data
        y_true_reg = np.array([1.5, 2.0, 3.0, 4.0, 5.0])
        y_pred_reg = np.array([1.7, 2.1, 2.9, 4.2, 4.8])

        analyzer = ResultsAnalyzer(task_type='regression')
        metrics = analyzer.calculate_metrics(
            y_true_reg, y_pred_reg,
            self.experiment_name
        )

        # Check if all expected metrics are present
        expected_metrics = ['mse', 'mae', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

    def test_results_storage(self):
        """Test storage of results and training history."""
        analyzer = ResultsAnalyzer()
        analyzer.calculate_metrics(
            self.y_true, self.y_pred,
            self.experiment_name,
            self.train_history
        )

        # Check if results are stored
        self.assertIn(self.experiment_name, analyzer.results)
        self.assertIn(self.experiment_name, analyzer.training_histories)

    def test_get_results(self):
        """Test getting results as DataFrame."""
        analyzer = ResultsAnalyzer()

        # Test empty results
        empty_df = analyzer.get_results()
        self.assertTrue(empty_df.empty)

        # Add some results
        analyzer.calculate_metrics(
            self.y_true, self.y_pred,
            self.experiment_name,
            self.train_history
        )

        # Test non-empty results
        results_df = analyzer.get_results()
        self.assertFalse(results_df.empty)
        self.assertEqual(len(results_df), 1)
        self.assertEqual(results_df.index[0], self.experiment_name)

        # Test multiple experiments
        analyzer.calculate_metrics(
            self.y_true, self.y_pred,
            'test_experiment_2',
            self.train_history
        )

        results_df = analyzer.get_results()
        self.assertEqual(len(results_df), 2)
        self.assertTrue('test_experiment' in results_df.index)
        self.assertTrue('test_experiment_2' in results_df.index)

    def test_print_results(self):
        """Test printing results."""
        analyzer = ResultsAnalyzer()

        # Test empty results
        empty_df = analyzer.print_results()
        self.assertTrue(empty_df.empty)

        # Add some results
        analyzer.calculate_metrics(
            self.y_true, self.y_pred,
            self.experiment_name,
            self.train_history
        )

        # Test non-empty results
        results_df = analyzer.print_results()
        self.assertFalse(results_df.empty)
        self.assertEqual(len(results_df), 1)
        self.assertEqual(results_df.index[0], self.experiment_name)


if __name__ == '__main__':
    unittest.main()
