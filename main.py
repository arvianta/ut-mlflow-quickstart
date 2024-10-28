import mlflow
import yaml
import os
import pandas as pd

from train import main_train
from clean import clean_n_transform
from preprocess import preprocess_data


# Initiate MLflow
mlflow.set_tracking_uri("http://localhost:6969")
mlflow.set_experiment("credit_score_classification_testing")

# Function to log features into features.yaml
def log_features_to_yaml(data, dataset_name='Dataset', file_name='features.yaml', description=''):
    """
    Logs the feature names and basic info of a DataFrame into a features.yaml file.

    Parameters:
    - data: The DataFrame containing the features to log.
    - dataset_name: A string specifying the name/type of the dataset (e.g., "Original" or "Processed").
    - file_name: The yaml file name.
    - description: The description of the set of features (optional).
    """
    # Extract feature names and data types
    feature_data_types = data.dtypes.apply(str).to_dict()
    
    # Prepare the feature info for YAML
    feature_info = {
        dataset_name: {
            'dataset_name': dataset_name,
            'features': feature_data_types,
            'num_features': len(data.columns),
            'description': description
        }
    }
    
    # Load existing YAML data if features.yaml already exists
    try:
        with open(file_name, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        existing_data = {}

    # Append the new dataset feature info to the existing data
    existing_data.update(feature_info)

    # Write updated data back to YAML file
    with open(file_name, 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False)

    print(f"Feature information for {dataset_name} saved to {file_name}")

# Function to clean up the features.yaml from the current directory
def clean_yaml(file_name='features.yaml'):
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"{file_name} has been deleted.")
    else:
        print(f"{file_name} does not exist.")
        
# Load dataset
data_path = './data/train.csv'
df = pd.read_csv(data_path)

# Log initial features to yaml
log_features_to_yaml(df, "Initial Features")

df_cleaned, clean_data_path = clean_n_transform(df)

# Log engineered features to yaml
log_features_to_yaml(df_cleaned, "Cleaned Features")

X_train, X_test, Y_train, Y_test = preprocess_data(df_cleaned)

# Combine and log the preprocessed data used for training
train_df = pd.concat([X_train, Y_train], axis=1)
log_features_to_yaml(train_df, "Training Features")

# End any active MLflow runs
if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run(run_name="main-run") as main_run:
    # Log features used within the runs
    mlflow.log_artifact("features.yaml")
    
    # Log cleaning, preprocessing, and training code
    mlflow.log_artifact("clean.py")
    mlflow.log_artifact("preprocess.py")
    mlflow.log_artifact("train.py")
    mlflow.log_artifact("main.ipynb")

    main_train(X_train, Y_train, X_test, Y_test, tuning_epochs=10, final_epochs=50, batch_size=32)