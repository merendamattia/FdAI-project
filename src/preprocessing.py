import pandas as pd
import numpy as np
import logging
import os

def setup_logging():
    """
    Sets up the logging configuration for the preprocessing module.
    Adds color support for console logs if colorama is available.
    Returns a logger instance named 'preprocessing'.
    """

    # Add color support for console logs
    try:
        from colorama import init, Fore, Style
        init(autoreset=True)
        class ColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.MAGENTA
            }
            def format(self, record):
                levelname = record.levelname
                color = self.COLORS.get(levelname, "")
                # Color only the levelname part
                record.levelname = f"{color}{levelname}{Style.RESET_ALL}"
                message = super().format(record)
                # Restore original levelname to avoid side effects
                record.levelname = levelname
                return message
    except ImportError:
        ColorFormatter = None

    # Setup logger
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, 'preprocessing.log')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    console_handler = logging.StreamHandler()
    if ColorFormatter:
        console_handler.setFormatter(ColorFormatter('%(asctime)s [%(levelname)s] %(message)s'))
    else:
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    logger = logging.getLogger("preprocessing")
    return logger

def get_logger():
    """
    Returns a singleton logger instance for the preprocessing module.
    Ensures that the logger is initialized only once.
    """

    # Singleton logger getter
    if not hasattr(get_logger, "_logger"):
        get_logger._logger = setup_logging()
    return get_logger._logger

def read_datasets(dataset_dir):
    """
    Reads the training dataset from the specified directory.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """

    logger = get_logger()
    logger.info(f'Reading dataset from {dataset_dir}/train.csv')
    df_train = pd.read_csv(f'{dataset_dir}/train.csv')
    logger.info(f'Reading dataset from {dataset_dir}/test.csv')
    df_test = pd.read_csv(f'{dataset_dir}/test.csv')
    return df_train, df_test

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by replacing '?', 'nan', and 'NaN' with numpy NaN values.

    Args:
        dataset (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Cleaned dataset with standardized missing values.
    """

    logger = get_logger()
    logger.info('Cleaning dataset: replacing "?", "nan", "NaN" with np.nan')
    return dataset.replace(['?', 'nan', 'NaN'], np.nan)

def save_dataset(df, path):
    """
    Saves the given DataFrame to a CSV file at the specified path.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Destination file path.
    """

    logger = get_logger()
    logger.info(f'Saving dataset to {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def get_feature_types(df, target=None):
    """
    Returns lists of numeric and categorical feature names from the DataFrame.
    Optionally excludes the target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str, optional): Name of the target column to exclude.

    Returns:
        tuple: (numeric_features, categorical_features)
            numeric_features (list): List of numeric feature names.
            categorical_features (list): List of categorical feature names.
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target is not None:
        numeric_features = [col for col in numeric_features if col != target]
        categorical_features = [col for col in categorical_features if col != target]
    return numeric_features, categorical_features

def remove_nan_pycaret(df):
    return impute_pycaret(df, numeric_imputation='drop', categorical_imputation='drop')

def impute_with_mean_pycaret(df):
    return impute_pycaret(df, numeric_imputation='mean')

def impute_with_mode_pycaret(df):
    return impute_pycaret(df, numeric_imputation='mode')

def impute_with_median_pycaret(df):
    return impute_pycaret(df, numeric_imputation='median')

def impute_pycaret(df, numeric_imputation='mean', categorical_imputation='mode'):
    from pycaret.classification import ClassificationExperiment
    logger = get_logger()
    logger.info(f'Imputing NaN values using {numeric_imputation} strategy')
    claexp = ClassificationExperiment()
    claexp.setup(data=df,
                 imputation_type='simple',
                 numeric_imputation=numeric_imputation,
                 categorical_imputation=categorical_imputation,
                 session_id=123,
                 verbose=False
                )
    if categorical_imputation == 'drop':
        initial_rows = len(df)
        df_clean = claexp.dataset_transformed
        removed_rows = initial_rows - len(df_clean)
        percent_removed = (removed_rows / initial_rows * 100) if initial_rows > 0 else 0
        logger.info(f'Removed {removed_rows} (over {initial_rows}) rows with NaN values ({percent_removed:.2f}%)')
        return df_clean
    return claexp.dataset_transformed

def remove_outliers_pycaret(df, threshold=0.05):
        """
        Removes outliers from the DataFrame using PyCaret's outlier removal.
        Outliers are detected using the IQR method with the specified threshold.

        Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): The IQR multiplier to define outliers.

        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """
        from pycaret.classification import ClassificationExperiment
        logger = get_logger()
        claexp = ClassificationExperiment()
        claexp.setup(data=df,
                     remove_outliers=True,
                     outliers_method='iforest',
                     outliers_threshold=threshold,
                     session_id=123,
                     verbose=False
                     )
        df_clean = claexp.dataset_transformed
        removed = len(df) - len(df_clean)
        logger.info(f'Removed {removed} outliers from {len(df)} rows using threshold {threshold}')
        return df_clean.reset_index(drop=True)

def normalize_pycaret(df, method='zscore'):
    from pycaret.classification import ClassificationExperiment
    logger = get_logger()
    logger.info(f'Normalizing features using {method} method')
    claexp = ClassificationExperiment()
    claexp.setup(data=df,
            normalize=True,
            normalize_method=method,
            session_id=123,
            verbose=False
            )
    return claexp.dataset_transformed

def transform_pycaret(df, method='quantile'):
    from pycaret.classification import ClassificationExperiment
    logger = get_logger()
    logger.info(f'Transforming features using {method} method')
    claexp = ClassificationExperiment()
    claexp.setup(data=df,
            transformation=True,
            transformation_method=method,
            session_id=123,
            verbose=False
            )
    return claexp.dataset_transformed

def main():
    logger = get_logger()
    dataset_dirs = ['datasets/classification/census_income',
                    'datasets/classification/bank_marketing']
    logger.info('Starting preprocessing of datasets')
    logger.info(f'Available datasets: {dataset_dirs}')

    for dataset_dir in dataset_dirs:
        logger.info(f'Processing dataset: {dataset_dir}')
        df_train, df_test = read_datasets(dataset_dir)

        # Cleaning the dataset: replace '?', 'nan', 'NaN' with numpy.nan
        df_train = clean_dataset(df_train)
        df_test = clean_dataset(df_test)
        # save_dataset(df_train, f'{dataset_dir}/01_with_NaN/train.csv')
        # save_dataset(df_test, f'{dataset_dir}/01_with_NaN/test.csv')
        # logger.debug(f'Train set after imputation: {df_test.shape[0]} rows, {df_test.shape[1]} features')
        # logger.debug(f'Test set after imputation: {df_test.shape[0]} rows, {df_test.shape[1]} features')

        # Identifying numeric and categorical features
        logger.info('Identifying numeric and categorical features')
        numeric_features, categorical_features = get_feature_types(df_train)
        logger.info(f'Numeric features ({numeric_features.count}): {numeric_features}')
        logger.info(f'Categorical features ({categorical_features.count}): {categorical_features}')

        # Scenario 1: remove NaN values
        df_train_mod = remove_nan_pycaret(df_train)
        df_test_mod = remove_nan_pycaret(df_test)
        save_dataset(df_train_mod, f'{dataset_dir}/01_without_NaN/train.csv')
        save_dataset(df_test_mod, f'{dataset_dir}/01_without_NaN/test.csv')
        logger.debug(f'Train set after imputation: {df_train_mod.shape[0]} rows, {df_train_mod.shape[1]} features')
        logger.debug(f'Test set after imputation: {df_test_mod.shape[0]} rows, {df_test_mod.shape[1]} features')

        # Scenario 2: impute NaN values using mean
        df_train_mod = impute_with_mean_pycaret(df_train)
        df_test_mod = impute_with_mean_pycaret(df_test)
        save_dataset(df_train_mod, f'{dataset_dir}/02_imputed_mean/train.csv')
        save_dataset(df_test_mod, f'{dataset_dir}/02_imputed_mean/test.csv')
        logger.debug(f'Train set after imputation: {df_train_mod.shape[0]} rows, {df_train_mod.shape[1]} features')
        logger.debug(f'Test set after imputation: {df_test_mod.shape[0]} rows, {df_test_mod.shape[1]} features')

        # Scenario 3: impute NaN values using mode
        df_train_mod = impute_with_mode_pycaret(df_train)
        df_test_mod = impute_with_mode_pycaret(df_test)
        save_dataset(df_train_mod, f'{dataset_dir}/03_imputed_mode/train.csv')
        save_dataset(df_test_mod, f'{dataset_dir}/03_imputed_mode/test.csv')
        logger.debug(f'Train set after imputation: {df_train_mod.shape[0]} rows, {df_train_mod.shape[1]} features')
        logger.debug(f'Test set after imputation: {df_test_mod.shape[0]} rows, {df_test_mod.shape[1]} features')

        # Scenario 4: impute NaN values using median
        df_train_mod = impute_with_median_pycaret(df_train)
        df_test_mod = impute_with_median_pycaret(df_test)
        save_dataset(df_train_mod, f'{dataset_dir}/04_imputed_median/train.csv')
        save_dataset(df_test_mod, f'{dataset_dir}/04_imputed_median/test.csv')
        logger.debug(f'Train set after imputation: {df_train_mod.shape[0]} rows, {df_train_mod.shape[1]} features')
        logger.debug(f'Test set after imputation: {df_test_mod.shape[0]} rows, {df_test_mod.shape[1]} features')

        # Scenario 5: remove outliers with different thresholds
        thresholds = [0.01, 0.03, 0.05]
        logger.info(f'Removing outliers with thresholds: {thresholds}')
        for threshold in thresholds:
            df_train_no_outliers = remove_outliers_pycaret(df_train, threshold=threshold)
            df_test_no_outliers = remove_outliers_pycaret(df_test, threshold=threshold)
            save_dataset(df_train_no_outliers, f'{dataset_dir}/05_no_outliers_{threshold}/train.csv')
            save_dataset(df_test_no_outliers, f'{dataset_dir}/05_no_outliers_{threshold}/test.csv')
            logger.debug(f'Train set after outlier removal (th={threshold}): {df_train_no_outliers.shape[0]} rows, {df_train_no_outliers.shape[1]} features')
            logger.debug(f'Test set after outlier removal (th={threshold}): {df_test_no_outliers.shape[0]} rows, {df_test_no_outliers.shape[1]} features')

        # Scenario 6: normalize features
        df_train_norm = normalize_pycaret(df_train)
        df_test_norm = normalize_pycaret(df_test)
        save_dataset(df_train_norm, f'{dataset_dir}/06_normalized/train.csv')
        save_dataset(df_test_norm, f'{dataset_dir}/06_normalized/test.csv')
        logger.debug(f'Train set after normalization: {df_train_norm.shape[0]} rows, {df_train_norm.shape[1]} features')
        logger.debug(f'Test set after normalization: {df_test_norm.shape[0]} rows, {df_test_norm.shape[1]} features')

        # Scenario 7: transform features
        df_train_trans = transform_pycaret(df_train)
        df_test_trans = transform_pycaret(df_test)
        save_dataset(df_train_trans, f'{dataset_dir}/07_transformed/train.csv')
        save_dataset(df_test_trans, f'{dataset_dir}/07_transformed/test.csv')
        logger.debug(f'Train set after transformation: {df_train_trans.shape[0]} rows, {df_train_trans.shape[1]} features')
        logger.debug(f'Test set after transformation: {df_test_trans.shape[0]} rows, {df_test_trans.shape[1]} features')

if __name__ == "__main__":
    main()