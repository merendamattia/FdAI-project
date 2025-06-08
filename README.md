# Impact of Data Preprocessing on Neural Network Performance

This project explores how different data preprocessing techniques affect the performance of neural networks in both classification and regression tasks. The study compares various preprocessing scenarios (e.g., imputation, outlier removal, normalization, and transformation) and evaluates their impact on model accuracy, robustness, and convergence.

## Features

- **Automated Neural Networks**
  - Dynamic architecture determination
  - Configurable hidden layers and units
  - Batch normalization and dropout for regularization
  - Support for both classification and regression tasks

- **Robust Data Preprocessing**
  - Missing value imputation (mean, median, mode)
  - Outlier detection and removal
  - Feature scaling and normalization
  - Categorical variable encoding
  - Input validation and error checking

## Setup and Usage

1. **Create a Conda Environment**

   ```bash
   conda create --name neural-network-performance-by-data-quality python=3.11 && \
   conda activate neural-network-performance-by-data-quality
   ```

2. **Install Dependencies**

   ```bash
   make setup
   ```

3. **Run Tests**

   ```bash
   make test
   ```

4. **Run Experiments**

   ```bash
   make run-classify    # to run classification experiments
   make run-regress     # to run regression experiments
   ```

5. **[Optional] Run single experiment**

   ```bash
   python src/main.py --task-type classification --dataset census_income --target salary
   python src/main.py --task-type regression --dataset house_price --target price
   ```

   Available options:
   - `--task-type`: `classification` or `regression`
   - `--dataset`: Name of dataset folder under `datasets/{task-type}/`
   - `--target`: Name of target column
   - `--epochs`: Number of training epochs (default: 100)
   - `--batch-size`: Training batch size (default: 32)


## Notes

- The project uses scikit-learn for preprocessing utilities.
- The project uses PyCaret for some preprocessing and model comparison tasks.
