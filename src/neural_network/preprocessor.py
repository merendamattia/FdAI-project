"""
Data preprocessing utilities for neural network input preparation.
"""
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """Handles data preprocessing for neural network input.

    This class provides functionality for preparing data for both classification
    and regression tasks, including handling missing values, encoding categorical
    variables, and scaling numerical features.

    Attributes:
        task_type (str): Type of task ('classification' or 'regression')
        scalers (dict): Dictionary of fitted scalers
        encoders (dict): Dictionary of fitted encoders
        feature_columns (list): List of feature column names
        fill_values (dict): Dictionary of values used to fill missing data
    """

    def __init__(self, task_type='classification'):
        """Initialize the preprocessor.

        Args:
            task_type (str, optional): Type of task. Defaults to 'classification'
        """
        self.task_type = task_type
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.fill_values = {}

    def fit_transform(self, df, target_column):
        """Fit the preprocessor to training data and transform it.

        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target column

        Returns:
            tuple: (X_tensor, y_tensor) preprocessed feature and target tensors

        Raises:
            ValueError: If target column is not found or DataFrame is empty/invalid
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df_processed = df.copy()

        # Validate target column
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Split features and target
        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])

        if X.empty:
            raise ValueError("No feature columns found in dataset")

        # Save feature columns for later use
        self.feature_columns = X.columns.tolist()

        # Handle missing values
        X = self._handle_missing_values(X)

        # Process features based on task type
        if self.task_type == 'classification':
            X = self._process_classification_features(X)
            y = self._process_classification_target(y)
        else:
            X = self._process_regression_features(X)
            y = self._process_regression_target(y)

        # Final NaN check
        if torch.isnan(X).any() or torch.isnan(y).any():
            raise ValueError("NaN values detected after preprocessing")

        return X, y

    def transform(self, df, target_column):
        """Transform new data using fitted preprocessor.

        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of target column

        Returns:
            tuple: (X_tensor, y_tensor) preprocessed feature and target tensors
        """
        if not self.feature_columns:
            raise ValueError("Preprocessor must be fitted before transform")

        df_processed = df.copy()

        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])

        # Ensure columns match training set
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Handle missing values
        X = self._handle_missing_values(X, is_training=False)

        # Transform features based on task type
        if self.task_type == 'classification':
            X = self._transform_classification_features(X)
            y = self._transform_classification_target(y)
        else:
            X = self._transform_regression_features(X)
            y = self._transform_regression_target(y)

        # Final NaN check
        if torch.isnan(X).any() or torch.isnan(y).any():
            raise ValueError("NaN values detected after preprocessing")

        return X, y

    def _handle_missing_values(self, X, is_training=True):
        """Handle missing values in the dataset.

        Args:
            X (pd.DataFrame): Feature DataFrame
            is_training (bool, optional): Whether this is training data. Defaults to True

        Returns:
            pd.DataFrame: DataFrame with handled missing values

        Raises:
            ValueError: If all values in a column are NaN during training
        """
        if is_training:
            self.fill_values = {}

        for col in X.columns:
            if is_training:
                if X[col].isna().all():
                    # For columns with all NaN, use 0 as fill value
                    self.fill_values[col] = 0
                elif X[col].dtype == np.number or pd.to_numeric(X[col], errors='coerce').notna().any():
                    # For numeric columns, use median
                    valid_values = pd.to_numeric(X[col], errors='coerce').dropna()
                    self.fill_values[col] = valid_values.median() if not valid_values.empty else 0
                else:
                    # For categorical columns, use mode or 'UNKNOWN'
                    mode_value = X[col].mode()
                    self.fill_values[col] = mode_value.iloc[0] if not mode_value.empty else 'UNKNOWN'

            # Fill missing values
            fill_value = self.fill_values.get(col, 0)
            X[col] = X[col].fillna(fill_value)

        return X

    def _process_classification_features(self, X):
        """Process features for classification tasks.

        Args:
            X (pd.DataFrame): Feature DataFrame

        Returns:
            torch.Tensor: Processed feature tensor
        """
        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le

        # Convert to numpy for more stable scaling
        X_np = X.to_numpy().astype(np.float64)

        # Manual standardization for more precise scaling
        means = np.mean(X_np, axis=0, keepdims=True)
        stds = np.std(X_np, axis=0, keepdims=True, ddof=1)  # ddof=1 for sample standard deviation
        stds[stds == 0] = 1  # Prevent division by zero
        X_scaled = (X_np - means) / stds

        # Store parameters in scaler for transform
        scaler = StandardScaler()
        scaler.mean_ = means.ravel()
        scaler.scale_ = stds.ravel()
        self.scalers['features'] = scaler

        return torch.FloatTensor(X_scaled)

    def _transform_classification_features(self, X):
        """Transform features for classification using fitted encoders/scalers.

        Args:
            X (pd.DataFrame): Feature DataFrame

        Returns:
            torch.Tensor: Transformed feature tensor
        """
        # Apply categorical encoding
        for col in X.columns:
            if col in self.encoders:
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except ValueError:
                    X[col] = 0

        # Apply scaling
        X_scaled = self.scalers['features'].transform(X)
        return torch.FloatTensor(X_scaled)

    def _process_classification_target(self, y):
        """Process target variable for classification tasks.

        Args:
            y (pd.Series): Target series

        Returns:
            torch.Tensor: Processed target tensor
        """
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            self.encoders['target'] = le_target
        else:
            y_encoded = y.values

        return torch.LongTensor(y_encoded)

    def _transform_classification_target(self, y):
        """Transform target variable using fitted encoder.

        Args:
            y (pd.Series): Target series

        Returns:
            torch.Tensor: Transformed target tensor
        """
        if 'target' in self.encoders:
            try:
                y_encoded = self.encoders['target'].transform(y)
            except ValueError:
                y_encoded = np.zeros(len(y))
        else:
            y_encoded = y.values

        return torch.LongTensor(y_encoded)

    def _process_regression_features(self, X):
        """Process features for regression tasks.

        Args:
            X (pd.DataFrame): Feature DataFrame

        Returns:
            torch.Tensor: Processed feature tensor
        """
        # Convert all columns to numeric, fill errors with median
        numeric_X = pd.DataFrame(X.copy())
        for col in X.columns:
            try:
                numeric_X[col] = pd.to_numeric(X[col], errors='coerce')
            except (ValueError, TypeError):
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    le = LabelEncoder()
                    numeric_X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = le
                else:
                    numeric_X[col] = 0

        # Handle any remaining NaN values with median
        for col in numeric_X.columns:
            median_val = numeric_X[col].median()
            numeric_X[col] = numeric_X[col].fillna(median_val if not pd.isna(median_val) else 0)

        # Scale features using StandardScaler
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(numeric_X)
        self.scalers['features'] = scaler

        return torch.FloatTensor(X_scaled)

    def _transform_regression_features(self, X):
        """Transform features for regression using fitted scaler.

        Args:
            X (pd.DataFrame): Feature DataFrame

        Returns:
            torch.Tensor: Transformed feature tensor
        """
        # Convert and encode features as in training
        numeric_X = pd.DataFrame()
        for col in X.columns:
            try:
                numeric_X[col] = pd.to_numeric(X[col], errors='coerce')
            except (ValueError, TypeError):
                if col in self.encoders:
                    try:
                        numeric_X[col] = self.encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        numeric_X[col] = 0
                else:
                    numeric_X[col] = 0

        # Handle any remaining NaN values
        for col in numeric_X.columns:
            numeric_X[col] = numeric_X[col].fillna(self.fill_values.get(col, 0))

        # Apply scaling
        X_scaled = self.scalers['features'].transform(numeric_X)
        return torch.FloatTensor(X_scaled)

    def _process_regression_target(self, y):
        """Process target variable for regression tasks.

        Args:
            y (pd.Series): Target series

        Returns:
            torch.Tensor: Processed target tensor
        """
        # Ensure target is numeric
        y = pd.to_numeric(y, errors='coerce')

        # Handle any NaN values in target with median
        median_target = y.median()
        y = y.fillna(median_target if not pd.isna(median_target) else 0)

        return torch.FloatTensor(y.values).view(-1, 1)

    def _transform_regression_target(self, y):
        """Transform target variable for regression.

        Args:
            y (pd.Series): Target series

        Returns:
            torch.Tensor: Transformed target tensor
        """
        # Ensure target is numeric
        y = pd.to_numeric(y, errors='coerce')

        # Fill NaN values with median from training if available
        if 'target' in self.fill_values:
            y = y.fillna(self.fill_values['target'])
        else:
            # If no fill value is available, use the current median
            median_target = y.median()
            y = y.fillna(median_target if not pd.isna(median_target) else 0)

        return torch.FloatTensor(y.values).view(-1, 1)
