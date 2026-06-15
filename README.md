## End to End ML Project  

import dagshub
dagshub.init(repo_owner='abdullahhashmi01', repo_name='Heart_Disease_ML-', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)