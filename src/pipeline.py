
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd

from transform import get_dataset
from training import train_model
from inference import running_inference



def data_pipeline():

    models = {
    "LogisticRegression": {
        "model": LogisticRegression,
        "params": {"solver": "liblinear", "C": 1.0, "random_state": 42},
        "param_grids": {"solver":["liblinear"], "C":[0.1, 0.5, 1]}
    },
    "RandomForest": {
        "model": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": 42}, 
        "param_grids": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10] }
    },
    "GradientBoost": {
        "model": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate":0.1, "random_state": 42}, 
        "param_grids": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "learning_rate":[0.01, 0.05, 0.1, 0.2] }
    }
}

    X_train, X_val, y_train, y_val, X_test, X_test_raw = get_dataset()

    train_model(models, X_train, y_train, X_val, y_val)

    prediction = running_inference(X_test, X_test_raw)

    prediction.to_csv('prediction.csv', index=False)

    return prediction


if  __name__ == "__main__":
    data_pipeline()

