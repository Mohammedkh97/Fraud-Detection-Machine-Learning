# Fraud-Detection-Machine-Learning

This application focuses on exploratory data analysis (EDA) for fraud detection and evaluates the performance of the XGBoost model on the dataset. It aims to uncover patterns that provide deeper insight into the characteristics of fraudulent transactions.

## Project Structure

- `demo_NB.ipynb`: The original Jupyter Notebook containing the full Exploratory Data Analysis, Feature Engineering, and Model Training workflow.
- `01_eda.py`: Standalone Python script covering Phase 1 (EDA and data visualization).
- `02_feature_engineering_and_training.py`: Standalone Python script covering Phase 2 (Feature Engineering and training Logistic Regression, Random Forest, and XGBoost models).
- `03_save_model.py`: Standalone Python script covering Phase 3 (Saving the trained models and making predictions on single transactions).
- `app.py`: Streamlit web application that provides a user interface for running real-time fraud predictions using the trained XGBoost model.

## Setup

1. **Install dependencies**: Make sure you have all required packages installed by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset**: Ensure that the dataset `AIML Dataset.csv` is present in the root directory before running the data processing scripts or notebook.

## Running the Web App

To launch the Fraud Detection AI web interface, run the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser where you can input transaction details and receive real-time fraud predictions.

