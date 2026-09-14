import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Load your existing files
try:
    # Load the tuned models
    with open('tuned_churn_models_light.pkl', 'rb') as f:
        models_data = pickle.load(f)
        tuned_models = models_data['tuned_models']
    
    # Load evaluation results
    eval_results = pd.read_csv('model_evaluation_results.csv')
    
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

# Set up MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer_Churn_Prediction")

# Log each model with proper metric names
for model_name, model in tuned_models.items():
    try:
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log basic model info
            mlflow.log_param("model_type", model_name)
            
            # Find the correct row in evaluation results
            model_metrics = eval_results[eval_results['Model'] == model_name].iloc[0]
            
            # Log metrics (using correct column names from your CSV)
            mlflow.log_metric("accuracy", model_metrics['Accuracy'])
            mlflow.log_metric("precision", model_metrics['Precision'])
            mlflow.log_metric("recall", model_metrics['Recall'])
            mlflow.log_metric("f1_score", model_metrics['F1 Score'])
            
            # Handle ROC AUC (check for both possible column names)
            roc_auc_value = model_metrics.get('ROC AUC', model_metrics.get('ROC_AUC', None))
            if roc_auc_value is not None:
                mlflow.log_metric("roc_auc", float(roc_auc_value))
            
            # Special handling for different model types
            if 'LogisticRegression' in str(type(model)):
                mlflow.sklearn.log_model(model, "logistic_regression_model")
            elif 'RandomForest' in str(type(model)):
                mlflow.sklearn.log_model(model, "random_forest_model")
            elif 'LGBM' in str(type(model)):
                import mlflow.lightgbm
                mlflow.lightgbm.log_model(model, "lightgbm_model")
            
            # Log your original files as artifacts
            mlflow.log_artifact('tuned_churn_models_light.pkl')
            mlflow.log_artifact('model_evaluation_results.csv')
            mlflow.log_artifact('hyperparameter_tuning_results_light.csv')
            
            print(f"Successfully logged {model_name} to MLflow")
            
    except Exception as e:
        print(f"Error logging {model_name}: {str(e)}")
        continue

print("MLflow logging completed!")