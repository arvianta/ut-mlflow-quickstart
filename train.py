import mlflow
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Step 4: Modified tune_model function
def tune_model(X_train, Y_train):
    with mlflow.start_run(run_name="hyperparameter-tuning", nested=True) as tuning_run:
        mlflow.log_params({
            "param_1": "param_1",
            "param_2": "param_2",
            "param_3": "param_3"
        })
        
        tuner = ""
        
        return tuner

def evaluate_model(model, X_test, Y_test, run_id=None):
    """
    Evaluate the model and log metrics to MLflow
    """
    # Get predictions
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)
    
    # Calculate metrics
    class_report = classification_report(Y_test_classes, Y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(Y_test_classes, Y_pred_classes)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot
    confusion_matrix_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Calculate additional metrics
    metrics = {
        'test_accuracy': class_report['accuracy'],
        'test_macro_avg_precision': class_report['macro avg']['precision'],
        'test_macro_avg_recall': class_report['macro avg']['recall'],
        'test_macro_avg_f1': class_report['macro avg']['f1-score'],
        'test_weighted_avg_precision': class_report['weighted avg']['precision'],
        'test_weighted_avg_recall': class_report['weighted avg']['recall'],
        'test_weighted_avg_f1': class_report['weighted avg']['f1-score']
    }
    
    # Add per-class metrics
    for i in range(3):  # Assuming 3 classes
        metrics.update({
            f'test_class_{i}_precision': class_report[str(i)]['precision'],
            f'test_class_{i}_recall': class_report[str(i)]['recall'],
            f'test_class_{i}_f1': class_report[str(i)]['f1-score']
        })
    
    return metrics, confusion_matrix_path

def save_model_config(hyperparameters, metrics, model_dir='model_artifacts'):
    """
    Save model configuration and metrics to YAML file
    """
    os.makedirs(model_dir, exist_ok=True)
    
    config = {
        'hyperparameters': hyperparameters,
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_path = os.path.join(model_dir, 'model_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def train_model(X_train, Y_train, X_test, Y_test, tuner):
    # Get hyperparameters as dictionary
    hyperparameters = {
        "param_1": "param_1",
        "param_2": "param_2",
        "param_3": "param_3",
        "param_4": "param_4"
    }

    final_model_name = f"BEST_MODEL_d{hyperparameters['param_1']}_{hyperparameters['param_2']}_{hyperparameters['param_3']}_lr{hyperparameters['param_4']}"

    with mlflow.start_run(run_name=final_model_name, nested=True) as run:
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        
        # TODO: Train model, log training metrics
        model = ""

        # TODO: Evaluate the model, log evaluation metrics
        metrics, confusion_matrix_path = evaluate_model(model, X_test, Y_test, run.info.run_id)
        
        # Log evaluation metrics
        mlflow.log_metrics(metrics)
        
        # Log confusion matrix plot
        mlflow.log_artifact(confusion_matrix_path)
        
        # Save and log model configuration
        config_path = save_model_config(hyperparameters, metrics)
        mlflow.log_artifact(config_path)
        
        # TODO: Save model locally and log it to MLflow
        temp_model_path = 'temp_model.h5/pkl'
        mlflow.log_artifact(temp_model_path, artifact_path="model")
        
        # Register the model
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "credit_score_classifier"
        )

        # Clean up temporary files
        os.remove(confusion_matrix_path)
        os.remove(config_path)
        os.remove(temp_model_path)

    return model, metrics, hyperparameters

def main_train(X_train, Y_train, X_test, Y_test):
    # Tune the model
    tuner = tune_model(X_train, Y_train)
    
    # Train and evaluate the model using the best hyperparameters
    model, metrics, hyperparameters = train_model(
        X_train, Y_train, 
        X_test, Y_test, 
        tuner
    )

    # Save the model locally
    model_dir = 'model_artifacts'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the Keras model
    model.save(os.path.join(model_dir, 'credit_score_model_tuned.h5'))
    
    print(f"\nModel training completed!")
    print(f"Model and configuration saved in: {model_dir}")
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    return model, metrics, hyperparameters

if __name__ == '__main__':
    pass