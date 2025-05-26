import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import json
from pathlib import Path
import pickle

NUM_EPOCHS = 20  # Default number of epochs for training

def setup_logging():
    """
    Sets up the logging configuration for the regression module with colored logs.
    """
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
                record.levelname = f"{color}{levelname}{Style.RESET_ALL}"
                message = super().format(record)
                record.levelname = levelname
                return message
    except ImportError:
        ColorFormatter = None

    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, 'regression.log')

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    console_handler = logging.StreamHandler()
    if ColorFormatter:
        console_handler.setFormatter(ColorFormatter('%(asctime)s [%(levelname)s] %(message)s'))
    else:
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    logger = logging.getLogger("regression")
    return logger

logging = setup_logging()

class AutoNeuralNetwork(nn.Module):
    """Neural network with automatic architecture for regression tasks."""

    def __init__(self, input_size):
        super(AutoNeuralNetwork, self).__init__()

        # Compute hidden layer sizes based on input size
        hidden1_size = max(64, input_size * 2)
        hidden2_size = max(32, input_size)
        hidden3_size = max(16, input_size // 2)

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden1_size, hidden2_size),
            nn.BatchNorm1d(hidden2_size),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden2_size, hidden3_size),
            nn.BatchNorm1d(hidden3_size),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden3_size, 1)  # Output layer for regression
        )

        # Xavier initialization for weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)

class DataPreprocessor:
    """Class for automatic data preprocessing."""

    def __init__(self):
        self.scalers = {}
        self.feature_columns = None

    def fit_transform(self, df, target_column):
        """
        Preprocess the training set: handle missing values and normalize features.
        """
        df_processed = df.copy()

        # Separate features and target
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])

        # Save feature names
        self.feature_columns = X.columns.tolist()

        # Handle missing values
        X = X.fillna(X.mean())

        # Normalize numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler

        return torch.FloatTensor(X_scaled), torch.FloatTensor(y.values).view(-1, 1)

    def transform(self, df, target_column):
        """
        Preprocess the test set using the same transformations as the training set.
        """
        df_processed = df.copy()

        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])

        # Ensure columns match training set
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Handle missing values
        X = X.fillna(0)

        # Normalize
        X_scaled = self.scalers['features'].transform(X)

        return torch.FloatTensor(X_scaled), torch.FloatTensor(y.values).view(-1, 1)

class ModelTrainer:
    """Class for training neural network models."""

    def __init__(self, device='cpu'):
        self.device = device

    def train_model(self, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32):
        """
        Train a neural network model and return the model, predictions, and training history.
        """
        # Model parameters
        input_size = X_train.shape[1]

        # Create model
        model = AutoNeuralNetwork(input_size).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training history
        train_losses = []

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(self.device)).cpu()

        return model, test_outputs, train_losses

class ResultsAnalyzer:
    """Class for analyzing and visualizing experiment results."""

    def __init__(self):
        self.results = {}
        self.training_histories = {}

    def calculate_metrics(self, y_true, y_pred, experiment_name, train_losses=None):
        """
        Compute evaluation metrics and store results and training history.
        """
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),  # Convert to native float
            'mae': float(mean_absolute_error(y_true, y_pred)),  # Convert to native float
            'r2': float(r2_score(y_true, y_pred))  # Convert to native float
        }

        self.results[experiment_name] = metrics

        # Save training history
        if train_losses is not None:
            self.training_histories[experiment_name] = {'losses': train_losses}

        return metrics

    def print_results(self):
        """
        Print the results of all experiments in a table and highlight the best model.
        """
        print("\n" + "="*80)
        print("FINAL RESULTS OF EXPERIMENTS")
        print("="*80)

        # Create DataFrame for better visualization
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)

        print(df_results.to_string())

        # Find the best model
        best_model = df_results['r2'].idxmax()
        best_r2 = df_results.loc[best_model, 'r2']

        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   R2-Score: {best_r2:.4f}")

        return df_results

    def plot_training_curves(self, output_dir):
        """
        Plot training loss curves for each experiment and save the plot.
        """
        if not self.training_histories:
            logging.warning("⚠️ No training history available for plots")
            return

        plt.figure(figsize=(10, 6))
        for exp_name, history in self.training_histories.items():
            epochs = range(1, len(history['losses']) + 1)
            plt.plot(epochs, history['losses'], label=exp_name)

        plt.title('Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Ensure y-axis values are displayed correctly
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

        plt.tight_layout()
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(os.path.join(img_dir, "training_curves.png"))
        plt.close()

    def plot_metrics_comparison(self, output_dir):
        """
        Plot a combined grid of bar charts for MSE, MAE, and R2 metrics across all experiments.
        """
        if not self.results:
            logging.warning("⚠️ No results available for plotting metrics comparison")
            return

        df_results = pd.DataFrame(self.results).T
        metrics = ['mse', 'mae', 'r2']
        colors = ['skyblue', 'lightcoral', 'lightgreen']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        fig.suptitle('Metrics Comparison Across Experiments', fontsize=16, fontweight='bold')

        for ax, metric, color in zip(axes, metrics, colors):
            values = df_results[metric].values

            if metric == 'r2':
                # Convert R2 to percentage
                values = values * 100
                min_value = max(min(values) - 2, 0)  # Ensure minimum is not below 0%
                max_value = max(values) * 1.02  # Add 2% padding for R2
                ax.set_ylim(min_value, max_value)
            else:
                max_value = max(values) * 2  # Add 2%0 padding for MSE and MAE
                ax.set_yscale('log')  # Set y-axis to logarithmic scale
                ax.set_ylim(1e-1, max_value)

            bars = ax.bar(
                df_results.index,
                values,
                color=color,
                alpha=0.7,
                edgecolor='black'
            )
            ax.set_title(metric.upper(), fontsize=14, fontweight='bold')
            ax.set_xlabel('Experiments')
            ax.set_ylabel(f'{metric.upper()} {"(%)" if metric == "r2" else ""}')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            # Add values on top of bars
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.01 if metric != 'r2' else value + 0.2,
                    f'{value:.1f}' if metric != 'mse' else f'{value:.0f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

        # Ensure the output directory exists
        img_dir = os.path.join(output_dir, "img")
        os.makedirs(img_dir, exist_ok=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
        plt.savefig(os.path.join(img_dir, "metrics_comparison.png"))
        plt.close()

def save_model_and_data(model, preprocessor, experiment_name, model_folder, train_losses=None):
    """
    Save the trained model, preprocessor, and training history.

    Args:
        model: The trained PyTorch model
        preprocessor: The DataPreprocessor object with fitted transformations
        experiment_name: Name of the experiment
        model_folder: Folder where to save the models
        train_losses: List of training losses (optional)
    """
    os.makedirs(model_folder, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_folder, f"{experiment_name}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': model.layers[0].in_features
        }
    }, model_path)

    # Save the preprocessor
    preprocessor_path = os.path.join(model_folder, f"{experiment_name}_preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    # Save the training history
    if train_losses is not None:
        history_path = os.path.join(model_folder, f"{experiment_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump({'losses': train_losses}, f, indent=2)

    logging.info(f"   ✅ Model saved to: {model_path}")
    logging.info(f"   ✅ Preprocessor saved to: {preprocessor_path}")
    if train_losses is not None:
        logging.info(f"   ✅ Training history saved to: {history_path}")

def load_model_and_data(experiment_name, model_folder, device):
    """
    Load the saved model, preprocessor, and training history.

    Args:
        experiment_name: Name of the experiment
        model_folder: Folder where the models are saved
        device: PyTorch device (cpu/cuda)

    Returns:
        tuple: (model, preprocessor, train_losses) or (None, None, None) if not found
    """
    model_path = os.path.join(model_folder, f"{experiment_name}_model.pth")
    preprocessor_path = os.path.join(model_folder, f"{experiment_name}_preprocessor.pkl")
    history_path = os.path.join(model_folder, f"{experiment_name}_history.json")

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None, None

    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model_info = checkpoint['model_architecture']

        model = AutoNeuralNetwork(input_size=model_info['input_size']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load the preprocessor
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        # Load the training history
        train_losses = None
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                train_losses = history.get('losses', [])

        logging.info(f"   ✅ Model loaded from: {model_path}")
        logging.info(f"   ✅ Preprocessor loaded from: {preprocessor_path}")
        if train_losses is not None:
            logging.info(f"   ✅ Training history loaded from: {history_path}")

        return model, preprocessor, train_losses

    except Exception as e:
        logging.error(f"   ❌ Loading error: {str(e)}")
        return None, None, None

def run_regression_study(dataset_folder, target_column):
    """
    Main function for running regression experiments.
    """
    root_folder = "datasets/"
    results_folder = "results/" + dataset_folder
    model_folder = "model/" + dataset_folder + "epochs_" + str(NUM_EPOCHS) + "/"

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    dataset_pairs = read_dataset_pairs(root_folder + dataset_folder)

    device = setup_device()
    trainer = ModelTrainer(device)
    analyzer = ResultsAnalyzer()

    for i, (train_path, test_path) in enumerate(dataset_pairs):
        experiment_name = extract_experiment_name(train_path)
        print("\n")
        logging.info(f"📊 EXPERIMENT {i+1}/{len(dataset_pairs)}: {experiment_name}")
        logging.info("-" * 50)

        try:
            # Check if model and history already exist
            model, preprocessor, train_losses = load_model_and_data(experiment_name, model_folder, device)

            if model is not None and preprocessor is not None:
                logging.info("🔄 Using existing model and preprocessor...")
                # Load and preprocess test data
                test_df = pd.read_csv(test_path)
                X_test, y_test = preprocessor.transform(test_df, target_column)

                # Generate predictions
                logging.info("🎯 Generating predictions...")
                model.eval()
                with torch.no_grad():
                    predictions = model(X_test.to(device)).cpu()

            else:
                logging.info("🎯 Training new model...")
                # Load data
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                preprocessor = DataPreprocessor()
                X_train, y_train = preprocessor.fit_transform(train_df, target_column)
                X_test, y_test = preprocessor.transform(test_df, target_column)

                # Train model
                model, predictions, train_losses = trainer.train_model(
                    X_train, y_train, X_test, y_test, num_epochs=NUM_EPOCHS
                )

                # Save model, preprocessor, and training history
                save_model_and_data(model, preprocessor, experiment_name, model_folder, train_losses)

            # Evaluate
            metrics = analyzer.calculate_metrics(
                y_test.numpy(), predictions.numpy(), experiment_name, train_losses
            )

            logging.info(f"   MSE: {metrics['mse']:.4f}")
            logging.info(f"   R2-Score: {metrics['r2']:.4f}")

        except Exception as e:
            logging.error(f"❌ Error in experiment {experiment_name}: {str(e)}")
            continue

    # Final analysis
    df_results = analyzer.print_results()
    analyzer.plot_training_curves(results_folder)
    analyzer.plot_metrics_comparison(results_folder)

    # Save results to JSON
    results_file = os.path.join(results_folder, "results.json")
    with open(results_file, "w") as f:
        # Ensure all values in results are JSON serializable
        json.dump({k: {metric: float(v) for metric, v in metrics.items()}
                   for k, metrics in analyzer.results.items()}, f, indent=4)
    logging.info(f"Results saved to {results_file}")

    return df_results

def read_dataset_pairs(root_folder):
    """
    Generate a list of dataset file pairs from the given root folder.
    For each subfolder under the root folder, if both 'train.csv'
    and 'test.csv' exist, their paths are added as a tuple to the list.

    Args:
        root_folder (str): The path to the dataset root folder.

    Returns:
        list: A list of tuples (train_file_path, test_file_path).
    """
    root = Path(root_folder)
    dataset_pairs = []

    for subfolder in sorted(root.iterdir()):
        if subfolder.is_dir():
            train_file = subfolder / "train.csv"
            test_file = subfolder / "test.csv"
            if train_file.exists() and test_file.exists():
                dataset_pairs.append((str(train_file), str(test_file)))
            else:
                logging.warning(f"Warning: One or both files missing in folder {subfolder.name}")

    logging.info(f"Found {len(dataset_pairs)} dataset pairs for comparison.")
    logging.info("Dataset pairs:")
    for i, (train_path, test_path) in enumerate(dataset_pairs):
        logging.info(f"{i+1}: {train_path} <-> {test_path}")
    logging.info("=" * 60)

    return dataset_pairs

def setup_device():
    """Automatically configure the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("💻 Using CPU")
    return device

def extract_experiment_name(path):
    """
    Extract the experiment name from the dataset path (parent directory name).
    """
    path_obj = Path(path)
    return path_obj.parent.name

if __name__ == "__main__":
    run_regression_study(dataset_folder="regression/bike_sharing/", target_column="cnt")
    # run_regression_study(dataset_folder="regression/house_prices/", target_column="price")
