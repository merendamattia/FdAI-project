import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images

IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

class AutoNeuralNetwork(nn.Module):
    """Neural network with automatic architecture based on the number of features."""

    def __init__(self, input_size, num_classes, task_type='classification'):
        super(AutoNeuralNetwork, self).__init__()
        self.task_type = task_type

        # Compute hidden layer sizes based on input size
        hidden1_size = max(64, input_size * 2)
        hidden2_size = max(32, input_size)
        hidden3_size = max(16, input_size // 2)

        self.layers = nn.ModuleList([
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

            nn.Linear(hidden3_size, num_classes)
        ])

        # Xavier initialization for weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DataPreprocessor:
    """Class for automatic data preprocessing."""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None

    def fit_transform(self, df, target_column):
        """
        Preprocess the training set: handle missing values, encode categorical variables,
        normalize features, and encode the target.
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
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])

        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le

        # Normalize numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler

        # Encode target
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            self.encoders['target'] = le_target
        else:
            y_encoded = y.values

        return torch.FloatTensor(X_scaled), torch.LongTensor(y_encoded)

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
        X = X.fillna(0)  # Use 0 as default for test set

        # Apply encoding
        for col in X.columns:
            if col in self.encoders:
                # Handle unseen categories
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except ValueError:
                    X[col] = 0  # Assign default value for unseen categories

        # Normalize
        X_scaled = self.scalers['features'].transform(X)

        # Encode target
        if 'target' in self.encoders:
            try:
                y_encoded = self.encoders['target'].transform(y)
            except ValueError:
                y_encoded = np.zeros(len(y))  # Default for unseen targets
        else:
            y_encoded = y.values

        return torch.FloatTensor(X_scaled), torch.LongTensor(y_encoded)

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
        num_classes = len(torch.unique(y_train))

        # Create model
        model = AutoNeuralNetwork(input_size, num_classes).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training history
        train_losses = []
        train_accuracies = []

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(self.device))
            _, test_predicted = torch.max(test_outputs, 1)
            test_predicted = test_predicted.cpu()

        return model, test_predicted, train_losses, train_accuracies

class ResultsAnalyzer:
    """Class for analyzing and visualizing experiment results."""

    def __init__(self):
        self.results = {}
        self.training_histories = {}

    def calculate_metrics(self, y_true, y_pred, experiment_name, train_losses=None, train_accuracies=None):
        """
        Compute evaluation metrics and store results and training history.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        self.results[experiment_name] = metrics

        # Save training history
        if train_losses is not None and train_accuracies is not None:
            self.training_histories[experiment_name] = {
                'losses': train_losses,
                'accuracies': train_accuracies
            }

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
        best_model = df_results['f1'].idxmax()
        best_f1 = df_results.loc[best_model, 'f1']

        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")

        return df_results

    def plot_comparison(self, df_results):
        """
        Create bar plots comparing the main metrics for all experiments.
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i//2, i%2]

            values = df_results[metric].values
            labels = df_results.index

            bars = ax.bar(range(len(labels)), values, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric.capitalize()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')

            # Add values on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "comparison.png"))
        plt.close(fig)

    def plot_training_curves(self):
        """
        Plot training loss and accuracy curves for each experiment.
        """
        if not self.training_histories:
            print("⚠️ No training history available for plots")
            return

        n_experiments = len(self.training_histories)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Distinct color palette
        colors = plt.cm.Set1(np.linspace(0, 1, n_experiments))

        # Loss plot
        ax1 = axes[0]
        for i, (exp_name, history) in enumerate(self.training_histories.items()):
            epochs = range(1, len(history['losses']) + 1)
            ax1.plot(epochs, history['losses'], color=colors[i], linewidth=2,
                    label=exp_name, marker='o', markersize=3, alpha=0.8)

        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Accuracy plot
        ax2 = axes[1]
        for i, (exp_name, history) in enumerate(self.training_histories.items()):
            epochs = range(1, len(history['accuracies']) + 1)
            ax2.plot(epochs, history['accuracies'], color=colors[i], linewidth=2,
                    label=exp_name, marker='s', markersize=3, alpha=0.8)

        ax2.set_title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "training_curves.png"))
        plt.close(fig)

    def plot_learning_comparison(self):
        """
        Plot a single loss and a single accuracy curve, comparing all experiments together.
        """
        if not self.training_histories:
            return

        n_experiments = len(self.training_histories)
        experiments = list(self.training_histories.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, n_experiments))

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Loss comparison
        ax_loss = axes[0]
        all_losses = [loss for h in self.training_histories.values() for loss in h['losses']]
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        for i, exp_name in enumerate(experiments):
            history = self.training_histories[exp_name]
            epochs = range(1, len(history['losses']) + 1)
            ax_loss.plot(epochs, history['losses'], label=exp_name, color=colors[i], linewidth=2, marker='o', markersize=4, alpha=0.85)
            ax_loss.text(epochs[-1], history['losses'][-1], f'{history["losses"][-1]:.4f}',
                         color=colors[i], fontsize=10, fontweight='bold', va='bottom', ha='right')
        ax_loss.set_title('Loss Comparison Across Experiments', fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')
        # Set y-limits to zoom in on the curves
        ax_loss.set_ylim(min_loss * 0.98, max_loss * 1.02)
        ax_loss.legend(loc='best')

        # Accuracy comparison
        ax_acc = axes[1]
        all_accs = [acc for h in self.training_histories.values() for acc in h['accuracies']]
        min_acc = min(all_accs)
        max_acc = max(all_accs)
        for i, exp_name in enumerate(experiments):
            history = self.training_histories[exp_name]
            epochs = range(1, len(history['accuracies']) + 1)
            ax_acc.plot(epochs, history['accuracies'], label=exp_name, color=colors[i], linewidth=2, marker='s', markersize=4, alpha=0.85)
            ax_acc.text(epochs[-1], history['accuracies'][-1], f'{history["accuracies"][-1]:.1f}%',
                        color=colors[i], fontsize=10, fontweight='bold', va='bottom', ha='right')
        ax_acc.set_title('Accuracy Comparison Across Experiments', fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.grid(True, alpha=0.3)
        # Set y-limits to zoom in on the curves
        ax_acc.set_ylim(max(min_acc - 2, 0), min(max_acc + 2, 100))
        ax_acc.legend(loc='best')

        plt.suptitle('Learning Curve Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(IMG_DIR, "learning_comparison.png"))
        plt.close(fig)

    def plot_convergence_analysis(self):
        """
        Analyze model convergence and stability.
        """
        if not self.training_histories:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Convergence speed (first 20 epochs)
        ax1 = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.training_histories)))

        for i, (exp_name, history) in enumerate(self.training_histories.items()):
            early_epochs = min(20, len(history['accuracies']))
            epochs = range(1, early_epochs + 1)
            ax1.plot(epochs, history['accuracies'][:early_epochs],
                    color=colors[i], linewidth=2, marker='o', label=exp_name)

        ax1.set_title('Convergence Speed (First 20 Epochs)', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Stability in last epochs
        ax2 = axes[0, 1]
        for i, (exp_name, history) in enumerate(self.training_histories.items()):
            last_epochs = min(20, len(history['accuracies']))
            epochs = range(len(history['accuracies']) - last_epochs + 1, len(history['accuracies']) + 1)
            ax2.plot(epochs, history['accuracies'][-last_epochs:],
                    color=colors[i], linewidth=2, marker='s', label=exp_name)

        ax2.set_title('Stability (Last 20 Epochs)', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Loss gradient (approximate derivative)
        ax3 = axes[1, 0]
        for i, (exp_name, history) in enumerate(self.training_histories.items()):
            losses = np.array(history['losses'])
            gradient = np.gradient(losses)
            epochs = range(1, len(gradient) + 1)
            ax3.plot(epochs, gradient, color=colors[i], linewidth=2, label=exp_name)

        ax3.set_title('Loss Gradient (Improvement Speed)', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Gradient')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # 4. Variance of last 10 epochs (stability)
        ax4 = axes[1, 1]
        stability_metrics = []
        exp_names = []

        for exp_name, history in self.training_histories.items():
            last_10_acc = history['accuracies'][-10:]
            variance = np.var(last_10_acc)
            stability_metrics.append(variance)
            exp_names.append(exp_name)

        bars = ax4.bar(range(len(exp_names)), stability_metrics,
                      color=colors[:len(exp_names)], alpha=0.7, edgecolor='black')
        ax4.set_title('Model Stability (Variance Last 10 Epochs)', fontweight='bold')
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Accuracy Variance')
        ax4.set_xticks(range(len(exp_names)))
        ax4.set_xticklabels(exp_names, rotation=45, ha='right')

        # Add values on bars
        for bar, value in zip(bars, stability_metrics):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_metrics)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        ax4.grid(axis='y', alpha=0.3)

        plt.suptitle('Convergence and Stability Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "convergence_analysis.png"))
        plt.close(fig)

    def plot_performance_heatmap(self, df_results):
        """
        Plot a heatmap of normalized performance metrics.
        """
        # Normalize results for better visualization
        df_normalized = df_results.copy()
        for col in df_normalized.columns:
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())

        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(df_normalized.T, annot=df_results.T, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Normalized Performance'},
                   xticklabels=df_results.index, yticklabels=df_results.columns)

        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Experiments')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "performance_heatmap.png"))
        plt.close()
        # Radar chart for multidimensional comparison
        self.plot_radar_chart(df_results)

    def plot_radar_chart(self, df_results):
        """
        Radar chart for multidimensional comparison of experiments.
        """
        # Prepare data for radar chart
        categories = list(df_results.columns)
        N = len(categories)

        # Compute angles for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set1(np.linspace(0, 1, len(df_results)))

        for i, (exp_name, row) in enumerate(df_results.iterrows()):
            values = row.tolist()
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=exp_name,
                   color=colors[i], markersize=6)
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.capitalize() for cat in categories])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)

        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.title('Multidimensional Performance Comparison\n(Radar Chart)',
                 size=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "radar_chart.png"))
        plt.close(fig)

    def plot_trends(self, df_results):
        """
        Plot trends of metrics across experiments.
        """
        plt.figure(figsize=(12, 8))

        experiments = list(df_results.index)

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = df_results[metric].values
            plt.plot(range(len(experiments)), values, marker='o', linewidth=2,
                     markersize=8, label=metric.capitalize())

        plt.title('Performance Trends Across Experiments', fontsize=14, fontweight='bold')
        plt.xlabel('Experiment')
        plt.ylabel('Score')
        plt.xticks(range(len(experiments)), experiments, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Adjust y-axis limits: start near the minimum value across metrics
        all_values = df_results[['accuracy', 'precision', 'recall', 'f1']].values.flatten()
        lower_lim = max(all_values.min() * 0.98, 0)
        upper_lim = min(all_values.max() * 1.02, 1)
        plt.ylim(lower_lim, upper_lim)

        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "trends.png"))
        plt.close()

def extract_experiment_name(path):
    """
    Extract the experiment name from the dataset path (parent directory name).
    """
    path_obj = Path(path)
    # Get the parent folder name (e.g., "02_without_NaN")
    return path_obj.parent.name

def setup_device():
    """Automatically configure the best available device (GPU or CPU)."""

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for fixed-size input
        torch.backends.cudnn.deterministic = False  # Allow faster, non-deterministic algorithms

        # Clear GPU memory cache
        torch.cuda.empty_cache()

    # # Check for Apple Silicon (MPS) support
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device('mps')
    #     print("🍎 Apple Silicon GPU (MPS) detected")

    # Fallback to CPU with optimizations
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")

        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())  # Use all available CPU cores
        print(f"   CPU threads used: {torch.get_num_threads()}")

    return device

def run_preprocessing_study(dataset_pairs, target_column):
    """
    Main function to run the preprocessing effectiveness study.
    """
    print("🚀 STARTING PREPROCESSING EFFECTIVENESS STUDY")
    print("="*60)

    device = setup_device()

    print(f"📱 Using device: {device}")

    trainer = ModelTrainer(device)
    analyzer = ResultsAnalyzer()

    for i, (train_path, test_path) in enumerate(dataset_pairs):
        experiment_name = extract_experiment_name(train_path)
        print(f"\n📊 EXPERIMENT {i+1}/{len(dataset_pairs)}: {experiment_name}")
        print("-" * 50)

        try:
            # Load data
            print("📥 Loading data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(f"   Training set: {train_df.shape}")
            print(f"   Test set: {test_df.shape}")

            # Preprocessing
            print("🔧 Preprocessing...")
            preprocessor = DataPreprocessor()
            X_train, y_train = preprocessor.fit_transform(train_df, target_column)
            X_test, y_test = preprocessor.transform(test_df, target_column)

            print(f"   Features: {X_train.shape[1]}")
            print(f"   Classes: {len(torch.unique(y_train))}")

            # Training
            print("🎯 Training the model...")
            model, predictions, train_losses, train_accuracies = trainer.train_model(
                X_train, y_train, X_test, y_test, num_epochs=100
            )

            # Evaluation
            print("📈 Calculating metrics...")
            metrics = analyzer.calculate_metrics(
                y_test.numpy(), predictions.numpy(), experiment_name,
                train_losses, train_accuracies
            )

            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")

        except Exception as e:
            print(f"❌ Error in experiment {experiment_name}: {str(e)}")
            continue

    # Final analysis
    print("\n" + "="*60)
    df_results = analyzer.print_results()
    analyzer.plot_comparison(df_results)
    analyzer.plot_trends(df_results)
    analyzer.plot_training_curves()
    analyzer.plot_learning_comparison()
    analyzer.plot_convergence_analysis()
    analyzer.plot_performance_heatmap(df_results)
    analyzer.plot_radar_chart(df_results)
    analyzer.plot_trends(df_results)

    return analyzer.results

if __name__ == "__main__":
    dataset_pairs = [
        ("datasets/classification/census_income/02_without_NaN/train.csv",
         "datasets/classification/census_income/02_without_NaN/test.csv"),
        ("datasets/classification/census_income/03_imputed_mean/train.csv",
         "datasets/classification/census_income/03_imputed_mean/test.csv"),
        ("datasets/classification/census_income/04_imputed_mode/train.csv",
         "datasets/classification/census_income/04_imputed_mode/test.csv"),
        ("datasets/classification/census_income/05_imputed_median/train.csv",
        "datasets/classification/census_income/05_imputed_median/test.csv"),
        ("datasets/classification/census_income/06_no_outliers_0.01/train.csv",
        "datasets/classification/census_income/06_no_outliers_0.01/test.csv"),
        ("datasets/classification/census_income/06_no_outliers_0.03/train.csv",
        "datasets/classification/census_income/06_no_outliers_0.03/test.csv"),
        ("datasets/classification/census_income/06_no_outliers_0.05/train.csv",
        "datasets/classification/census_income/06_no_outliers_0.05/test.csv"),
        ("datasets/classification/census_income/07_normalized/train.csv",
        "datasets/classification/census_income/07_normalized/test.csv"),
        ("datasets/classification/census_income/08_transformed/train.csv",
        "datasets/classification/census_income/08_transformed/test.csv"),
    ]

    # Target column name
    target_column = "salary"  # Change to your target column name

    # Run the study
    results = run_preprocessing_study(dataset_pairs, target_column)