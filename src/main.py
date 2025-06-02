"""
Main script for running neural network experiments.
"""
import os
import argparse
import pandas as pd
import torch
from typing import List, Tuple, Optional
from neural_network import (
    AutoClassifier,
    AutoRegressor,
    DataPreprocessor,
    ModelTrainer,
    ResultsAnalyzer,
    setup_logging,
    get_device
)

logger = setup_logging('main')


def read_dataset_pairs(dataset_folder: str) -> List[Tuple[str, str]]:
    """Read pairs of train/test datasets from the given folder.

    Args:
        dataset_folder: Path to dataset folder

    Returns:
        List of (train_path, test_path) tuples
    """
    dataset_pairs = []
    for root, _, files in os.walk(dataset_folder):
        if 'train.csv' in files and 'test.csv' in files:
            train_path = os.path.join(root, 'train.csv')
            test_path = os.path.join(root, 'test.csv')
            dataset_pairs.append((train_path, test_path))
    return sorted(dataset_pairs)


def extract_experiment_name(filepath: str) -> str:
    """Extract experiment name from filepath.

    Args:
        filepath: Path to dataset file

    Returns:
        Name of the experiment
    """
    return os.path.basename(os.path.dirname(filepath))


def save_model_and_data(
    model: torch.nn.Module,
    preprocessor: DataPreprocessor,
    experiment_name: str,
    model_folder: str
) -> None:
    """Save trained model and preprocessor.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        experiment_name: Name of experiment
        model_folder: Folder to save model
    """
    os.makedirs(model_folder, exist_ok=True)

    # Save model
    model_path = os.path.join(model_folder, f"{experiment_name}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.get_architecture_info()
    }, model_path)

    # Save preprocessor
    preprocessor_path = os.path.join(model_folder, f"{experiment_name}_preprocessor.pkl")
    torch.save(preprocessor, preprocessor_path)

    logger.info(f"Model and preprocessor saved in {model_folder}")


def run_experiment(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    """Run neural network experiments.

    Args:
        args: Command line arguments

    Returns:
        DataFrame containing experiment results, or None if all experiments failed
    """
    # Setup paths
    dataset_folder = os.path.join("datasets", args.task_type, args.dataset)
    results_folder = os.path.join("results", args.task_type, args.dataset)
    model_folder = os.path.join("model", args.task_type, args.dataset, f"epochs_{args.epochs}")

    if not os.path.exists(dataset_folder):
        logger.error(f"Dataset folder not found: {dataset_folder}")
        return None

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Setup components
    preprocessor = DataPreprocessor(task_type=args.task_type)
    trainer = ModelTrainer(device=device, task_type=args.task_type)
    analyzer = ResultsAnalyzer(task_type=args.task_type)

    # Get dataset pairs
    dataset_pairs = read_dataset_pairs(dataset_folder)
    if not dataset_pairs:
        logger.error(f"No dataset pairs found in {dataset_folder}")
        return None

    logger.info(f"Found {len(dataset_pairs)} dataset pairs")
    successful_experiments = 0

    # Run experiments
    for i, (train_path, test_path) in enumerate(dataset_pairs, 1):
        experiment_name = extract_experiment_name(train_path)
        logger.info(f"Experiment {i}/{len(dataset_pairs)}: {experiment_name}")
        logger.info("-" * 50)

        try:
            # Load and preprocess data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if args.target not in train_df.columns or args.target not in test_df.columns:
                logger.error(f"Target column '{args.target}' not found in dataset")
                continue

            X_train, y_train = preprocessor.fit_transform(train_df, args.target)
            X_test, y_test = preprocessor.transform(test_df, args.target)

            # Create and train model
            input_size = X_train.shape[1]
            if args.task_type == 'classification':
                num_classes = len(torch.unique(y_train))
                model = AutoClassifier(input_size, num_classes).to(device)
            else:
                model = AutoRegressor(input_size).to(device)

            # Train model
            model, predictions, losses, metrics = trainer.train_model(
                model, X_train, y_train, X_test, y_test,
                num_epochs=args.epochs,
                batch_size=args.batch_size
            )

            # Save model and preprocessor
            save_model_and_data(model, preprocessor, experiment_name, model_folder)

            # Calculate and store metrics
            analyzer.calculate_metrics(
                y_test.cpu().numpy(),
                predictions.cpu().numpy(),
                experiment_name,
                {'losses': losses, 'metrics': metrics}
            )

            successful_experiments += 1

        except Exception as e:
            logger.error(f"Error in experiment {experiment_name}: {str(e)}")
            continue

    if successful_experiments == 0:
        logger.error("All experiments failed")
        return None

    # Print final results
    results_df = analyzer.get_results()
    logger.info("\nFinal Results:")
    logger.info(results_df)

    return results_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run neural network experiments')
    parser.add_argument('--task-type', choices=['classification', 'regression'],
                       required=True, help='Type of task')
    parser.add_argument('--dataset', required=True,
                       help='Name of dataset folder')
    parser.add_argument('--target', required=True,
                       help='Name of target column')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')

    args = parser.parse_args()
    try:
        results = run_experiment(args)
        if results is None:
            exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
