# Impact of Data Preprocessing on Neural Network Performance

This project explores how different data preprocessing techniques affect the performance of neural networks in both classification and regression tasks. The study compares various preprocessing scenarios (e.g., imputation, outlier removal, normalization, and transformation) and evaluates their impact on model accuracy, robustness, and convergence.

## Project Structure

- **src/**: Contains the source code for classification (`ann-classification.py`) and regression (`ann-regression.py`) studies.
- **datasets/**: Organized folders for classification and regression datasets.
- **model/**: Saved models resulting from the experiments.
- **results/**: Experiment outcomes, including logs and visualizations.

## Setup and Experiment Replication

Follow these steps to replicate the experiments:

1. **Create a Conda Environment**

   Create and activate a new Conda environment for the project. For example:
   ```bash
   conda create --name neural-network-performance-by-data-quality python=3.11
   conda activate neural-network-performance-by-data-quality
   ```

2. **Install Dependencies**

   Run the following command inside the conda environment to set up the required dependencies:
   ```bash
   make setup
   ```

3. **Run the Experiments**

   - To run the classification experiments, execute:
     ```bash
     cd src && \
     python ann-classification.py
     ```
   - To run the regression experiments, execute:
     ```bash
     cd src && \
     python ann-regression.py
     ```

## Notes

- The project uses PyCaret for some preprocessing and model comparison tasks.
