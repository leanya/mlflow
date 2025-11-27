#### A machine learning experimentation project with MLflow
A machine learning experimentation project that tracks runs, tunes parameters, compares models, and manages model version using MLflow

#### Project Components  
- Data Transformation
    - Handle missing data using interpolation
    - Ensure data formats are consistent
    - Standardise features
- Data Training
    - Train multiple baseline models (e.g. Logistic Regression, Random Forest, Gradient Boosting)
    - Track experiment runs with MLflow. Information such as model artifacts, model parameters, output metrics (e.g. accuracy)  are logged 
    - Register each trained model in MLflow registry
    - The top performing models are identified and promoted to the Data Evaluation stage where the hyperparameters are tuned
    - Within MLflow registry, models are grouped by stage for easy comparison
        - Data Training stage
        - Data Evaluation stage
- Data Inference 
    - Identify the best performing tuned model as the final model
    - Apply final model to the test dataset to generate the predictions
- Data Pipeline
    - Runs the data pipeline including data transformation, training, inference

#### Running the project
- Start the MLflow tracking server with  
 ```mlflow server --backend-store-uri="mysql+pymysql://username@hostname:port/database" --default-artifact-root=s3://your-bucket --host=0.0.0.0 --port=5000```
- Runs the data pipeline script
- Model Registry 

#### References 
https://www.kaggle.com/competitions/titanic/data
https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/
https://mlflow.org/docs/latest/ml/tracking/tutorials/remote-server/