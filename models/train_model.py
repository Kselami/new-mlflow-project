import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
import sys

def train_model():
    # Vérifier si on est en environnement CI
    is_ci = os.getenv('CI') == 'true'
    
    # Construire le chemin absolu vers les données
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'sample_data.csv')
    
    # Charger les données
    data = pd.read_csv(data_path)
    
    # Utiliser un subset plus petit en CI pour accélérer l'entraînement
    if is_ci and len(data) > 100:
        data = data.sample(100, random_state=42)
    
    # Séparer features et target
    X = data[['feature1', 'feature2', 'feature3']]
    y = data['target']
    
    # Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Démarrer une expérience MLflow
    mlflow.set_experiment("MLOps-CI-CD-Experiment")
    
    with mlflow.start_run():
        # Entraîner le modèle
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Faire des prédictions
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Logger les paramètres
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("is_ci", is_ci)
        mlflow.log_param("dataset_size", len(data))
        
        # Logger les métriques
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Logger le modèle
        mlflow.sklearn.log_model(model, "model")
        
        # Enregistrer les métriques dans un fichier pour le CI/CD
        output_dir = os.path.join(base_dir, 'mlflow_output')
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        
        with open(metrics_path, "w") as f:
            f.write(f"MSE: {mse}\n")
            f.write(f"R2: {r2}\n")
        
        print(f"Modèle entraîné avec MSE: {mse}, R2: {r2}")
        
    return model, mse, r2

if __name__ == "__main__":
    train_model()