import pandas as pd
import mlflow.sklearn
import numpy as np

def load_model(run_id):
    model_path = f"../mlruns/0/{run_id}/artifacts/model"
    model = mlflow.sklearn.load_model(model_path)
    return model

def make_prediction(model, features):
    prediction = model.predict(features)
    return prediction

if __name__ == "__main__":
    # Exemple d'utilisation
    sample_features = np.array([[2.5, 4.8, 6.1]])
    run_id = "your_run_id_here"  # À remplacer par un ID d'exécution réel
    
    model = load_model(run_id)
    prediction = make_prediction(model, sample_features)
    print(f"Prediction: {prediction[0]}")