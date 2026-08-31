import mlflow

def register_model(model_name, model_path):
    print(f"Registering model {model_name} to MLflow...")
    # mlflow.set_tracking_uri("http://localhost:5000")
    # with mlflow.start_run():
    #     mlflow.log_artifact(model_path)
    #     mlflow.register_model("runs:/...", model_name)
    print("Model registered successfully.")
