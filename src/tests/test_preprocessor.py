"""
Unit tests for data preprocessing functionality.
"""
import unittest
import pandas as pd
import numpy as np
import torch
from neural_network.preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test cases."""
        # Create sample data for testing
        np.random.seed(42)

        # Classification dataset
        self.train_data_clf = pd.DataFrame({
            'num_feature1': [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, np.nan],
            'num_feature2': [-1.0, -2.0, np.nan, 1.0, 2.0, np.nan, 3.0, 4.0],
            'cat_feature1': ['A', 'B', 'A', 'C', np.nan, 'B', 'C', 'A'],
            'cat_feature2': ['X', np.nan, 'Y', 'Z', 'X', 'Y', np.nan, 'Z'],
            'target': [0, 1, 0, 1, 1, 0, 1, 0]
        })

        self.test_data_clf = pd.DataFrame({
            'num_feature1': [2.0, np.nan, 4.0, 6.0],
            'num_feature2': [1.0, 2.0, np.nan, 3.0],
            'cat_feature1': ['B', 'D', 'A', 'C'],
            'cat_feature2': ['Y', 'X', 'Z', 'W'],
            'target': [1, 0, 1, 0]
        })

        # Regression dataset
        self.train_data_reg = pd.DataFrame({
            'num_feature1': [1.0, 2.0, 3.0, np.nan, 5.0],
            'num_feature2': [-1.0, -2.0, np.nan, 1.0, 2.0],
            'target': [10.5, 20.3, 15.7, 25.1, 30.0]
        })

        self.test_data_reg = pd.DataFrame({
            'num_feature1': [2.0, np.nan, 4.0],
            'num_feature2': [1.0, 2.0, np.nan],
            'target': [12.8, 18.5, 22.3]
        })

    def test_classification_preprocessing(self):
        """Test preprocessing for classification task."""
        preprocessor = DataPreprocessor(task_type='classification')
        X_train, y_train = preprocessor.fit_transform(self.train_data_clf, 'target')

        # Check output types
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)

        # Check shapes
        self.assertEqual(X_train.shape[0], len(self.train_data_clf))
        self.assertEqual(X_train.shape[1], len(self.train_data_clf.columns) - 1)
        self.assertEqual(y_train.shape[0], len(self.train_data_clf))

        # Check if missing values are handled
        self.assertFalse(torch.isnan(X_train).any())

        # Test standardization
        self.assertAlmostEqual(X_train[:, 0].mean().item(), 0, places=5)
        self.assertAlmostEqual(X_train[:, 0].std().item(), 1, places=5)

        # Check categorical encoding
        unique_categories = len(torch.unique(X_train[:, 2]))  # cat_feature1
        self.assertGreater(unique_categories, 1)

        # Test transform on test data
        X_test, y_test = preprocessor.transform(self.test_data_clf, 'target')
        self.assertEqual(X_test.shape[1], X_train.shape[1])
        self.assertFalse(torch.isnan(X_test).any())

    def test_missing_target_column(self):
        """Test handling of missing target column."""
        preprocessor = DataPreprocessor()
        with self.assertRaises(ValueError):
            preprocessor.fit_transform(self.train_data_clf, 'non_existent_target')

    def test_feature_column_consistency(self):
        """Test consistency of feature columns between fit and transform."""
        preprocessor = DataPreprocessor()
        X_train, _ = preprocessor.fit_transform(self.train_data_clf, 'target')

        # Create test data with different column order
        cols = list(self.test_data_clf.columns)
        reordered_cols = cols[2:] + cols[:2]  # Shuffle columns
        reordered_test = self.test_data_clf[reordered_cols]
        X_test, _ = preprocessor.transform(reordered_test, 'target')

        self.assertEqual(X_test.shape[1], X_train.shape[1])

    def test_missing_value_handling(self):
        """Test handling of missing values."""
        preprocessor = DataPreprocessor()
        X_train, _ = preprocessor.fit_transform(self.train_data_clf, 'target')

        # Check if fill values are stored
        self.assertTrue(hasattr(preprocessor, 'fill_values'))
        self.assertGreater(len(preprocessor.fill_values), 0)

        # Test numeric and categorical filling
        num_col = 'num_feature1'
        cat_col = 'cat_feature1'

        self.assertIsNotNone(preprocessor.fill_values.get(num_col))
        self.assertIsNotNone(preprocessor.fill_values.get(cat_col))

        # Test filling in transform
        X_test, _ = preprocessor.transform(self.test_data_clf, 'target')
        self.assertFalse(torch.isnan(X_test).any())

    def test_data_types(self):
        """Test handling of different data types."""
        # Create data with various types
        mixed_data = pd.DataFrame({
            'int_col': [1, 2, np.nan, 4],
            'float_col': [1.5, np.nan, 3.5, 4.5],
            'str_col': ['a', 'b', np.nan, 'd'],
            'bool_col': [True, False, True, np.nan],
            'target': [0, 1, 0, 1]
        })

        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.fit_transform(mixed_data, 'target')

        self.assertFalse(torch.isnan(X_train).any())
        self.assertEqual(X_train.dtype, torch.float32)
        self.assertEqual(y_train.dtype, torch.long)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        preprocessor = DataPreprocessor()

        # Empty DataFrame
        empty_df = pd.DataFrame({'target': []})
        with self.assertRaises(Exception):
            preprocessor.fit_transform(empty_df, 'target')

        # Single column DataFrame
        single_col_df = pd.DataFrame({'target': [1, 2, 3]})
        with self.assertRaises(Exception):
            preprocessor.fit_transform(single_col_df, 'target')

        # All missing values in a column
        all_missing_df = pd.DataFrame({
            'feature': [np.nan, np.nan, np.nan],
            'target': [1, 2, 3]
        })
        X_train, _ = preprocessor.fit_transform(all_missing_df, 'target')
        self.assertFalse(torch.isnan(X_train).any())


if __name__ == '__main__':
    unittest.main()
