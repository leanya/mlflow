
# https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/quickstart/quickstart-sklearn/
# https://mlflow.org/docs/latest/ml/tracking/quickstart/

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from transform import get_dataset

# ---------------------------
# Utilities
# ---------------------------
def compute_metrics(actual, predicted):
   
   # only accuracy is used as the evaluation metrics for test dataset on Kaggle
   accuracy = accuracy_score(actual, predicted)
   precision = precision_score(actual, predicted)
   recall = recall_score(actual, predicted)
   f1 = f1_score(actual, predicted)
   
   return accuracy, precision, recall, f1 

def _helper_register_model(client, registered_name, model_name, run):
    # Register model in MLflow Registry

    try:
        client.create_registered_model(registered_name)
    except Exception:
        pass  # already exists

    mv = client.create_model_version(
        name=registered_name,
        source=f"runs:/{run.info.run_id}/{model_name}",
        run_id=run.info.run_id
        )

    client.set_model_version_tag(
        name=registered_name,
        version=mv.version,
        key="algorithm",
        value=model_name
        )
    client.set_model_version_tag(
        name=registered_name,
        version=mv.version,
        key="validation_status",
        value="pending"
        )

# Promote top models within the model registry
def select_top_models(client, 
                      experiment_name="Titanic_models", 
                      top_n=2, 
                      metric="train_accuracy", 
                      registered_name = "TitanicModel_training"):
    
    # search the runs and get the top_n runs 
    phase = registered_name.split('_')[1]
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.runName LIKE '{phase}_%'"
    )
    runs = runs.sort_values(by="start_time", ascending=False)
    runs_sorted = runs.sort_values(by= f"metrics.{metric}", ascending=False)
    top_runs = runs_sorted.head(top_n)
    
    # Update the model tags in the model registry 
    for _, row in top_runs.iterrows():
      run_id = row.run_id
      versions = client.search_model_versions(f"name='{registered_name}' and run_id='{run_id}'")
      if not versions:
         print(f"No model version found for run {run_id}")
         continue

      mv = versions[0]
      # update the validation status tag
      client.set_model_version_tag(
            name=registered_name ,
            version=mv.version,
            key="validation_status",
            value="approved"
         )

# ---------------------------
# Phase 1: Training
# ---------------------------
def run_training(models,  
                 X_train, 
                 y_train, 
                 client, 
                 experiment_name="Titanic_models", 
                 registered_name = "TitanicModel_training"):

   # Set our tracking server uri for logging
   mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

   # Create a new MLflow Experiment
   mlflow.set_experiment(experiment_name)
   
   for model_name, model_info in models.items():
      with mlflow.start_run(run_name = f"training_{model_name}") as run:

         # Log the parameters for the model
         for k, v in model_info["params"].items():
            mlflow.log_param(k, v)

         # Train and log our model, which inherits the parameters
         model = model_info["model"](**model_info["params"])
         model.fit(X_train, y_train)
         mlflow.sklearn.log_model(sk_model=model, 
                                  name=model_name, 
                                  input_example=X_train)

         # Evaluate the model on the training dataset and log metrics
         predictions = model.predict(X_train)
         (accuracy, precision, recall, f1) = compute_metrics(y_train, predictions)
         # Log the metrics 
         mlflow.log_metrics(
            metrics={
               "train_accuracy": accuracy,
               "train_precision": precision,
               "train_recall": recall,
               "train_f1": f1
            }
            )
         
         # Register baseline/training model with MLflow Model Registry    
         _helper_register_model(client, registered_name, model_name, run)

# ---------------------------
# Phase 2: Evaluation / Hyperparameter Tuning
# --------------------------- 
def hyperparameter_tuning(models, 
                          X_val, 
                          y_val, 
                          client,
                          experiment_name="Titanic_models", 
                          training_name="TitanicModel_training", 
                          registered_name = "TitanicModel_evaluation"):    
    

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    
    # extract the models from the model registry for evaluation/hyperparameter tuning
    approved_versions = []
    for mv in client.search_model_versions(f"name='{training_name}'"):
        if mv.tags.get("validation_status") == "approved":
            approved_versions.append(mv)
    
    for mv in approved_versions:

        model_name = mv.tags.get("algorithm") 
        spec = models[model_name]  # fetch spec with grids
        param_grid = spec.get("param_grids", {})
        base_model = spec["model"]
        params = spec["params"]
        
        model_instance = base_model(**params)
        grid = GridSearchCV(model_instance, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_val, y_val)

        # Log tuned model into evaluation phase
        with mlflow.start_run(run_name=f"evaluation_{model_name}") as run:
            
            mlflow.sklearn.log_model(sk_model = grid.best_estimator_,  
                                     name = model_name,
                                     input_example = X_val)

            mlflow.log_params(grid.best_params_)

            predictions = grid.best_estimator_.predict(X_val)
            (accuracy, precision, recall, f1) = compute_metrics(y_val, predictions)
            # These metrics will be linked to both the model and run
            mlflow.log_metrics(
               metrics={
                  "val_accuracy": accuracy,
                  "val_precision": precision,
                  "val_recall": recall,
                  "val_f1": f1
                  }
                  )
            # Register evaluation model in MLflow Model Registry    
            _helper_register_model(client, registered_name, model_name, run)

# ---------------------------
# Main Function
# ---------------------------

def train_model(models, X_train, y_train, X_val, y_val):
      
      client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

      run_training(models,  X_train, y_train, client)
   
      select_top_models(client)
   
      hyperparameter_tuning(models, X_val, y_val,client) 

      select_top_models(client, 
                        experiment_name="Titanic_models", 
                        top_n=1, 
                        metric="val_accuracy", 
                        registered_name = "TitanicModel_evaluation")


# if __name__ == "__main__":
   
#    models = {
#     "LogisticRegression": {
#         "model": LogisticRegression,
#         "params": {"solver": "liblinear", "C": 1.0, "random_state": 42},
#         "param_grids": {"solver":["liblinear"], "C":[0.1, 0.5, 1]}
#     },
#     "RandomForest": {
#         "model": RandomForestClassifier,
#         "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42}, 
#         "param_grids": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10] }
#     },
#     "GradientBoost": {
#         "model": GradientBoostingClassifier,
#         "params": {"n_estimators": 100, "max_depth": 5, "learning_rate":0.1, "random_state": 42}, 
#         "param_grids": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "learning_rate":[0.01, 0.05, 0.1, 0.2] }
#     }
# }


#    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
#    X_train, X_val, y_train, y_val, X_test = get_dataset()
   
#    run_training(models,  X_train, y_train, client)
   
#    select_top_models(client)
   
#    hyperparameter_tuning(models, X_val, y_val,client) 

#    select_top_models(client, 
#                      experiment_name="Titanic_models", 
#                      top_n=1, 
#                      metric="val_accuracy", 
#                      registered_name = "TitanicModel_evaluation")




   