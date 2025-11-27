import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import pandas as pd
from transform import get_dataset

# ---------------------------
# Final Phase: Inference
# ---------------------------

def inference(client, 
             X_test,
             X_test_raw,
             registered_name = "TitanicModel_evaluation"):
    
    # extract the models from the model registry for inference
    approved_versions = []
    for mv in client.search_model_versions(f"name='{registered_name}'"):
        if mv.tags.get("validation_status") == "approved":
            approved_versions.append(mv)
    if not approved_versions:
        raise ValueError("No approved model found!")
    
    mv = approved_versions[0]
    model_uri = f"models:/{registered_name}/{mv.version}"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X_test)
    
    pred_dict = {'PassengerId': X_test_raw["PassengerId"] , 'Survived': predictions*1 }
    pred_df = pd.DataFrame(data = pred_dict)

    return pred_df

# ---------------------------
# Main Function
# ---------------------------

def running_inference(X_test, X_test_raw):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    output = inference(client, X_test, X_test_raw)

    return output

# if __name__ == "__main__":

#     mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
#     _, _, _, _, X_test = get_dataset()
#     output = inference()