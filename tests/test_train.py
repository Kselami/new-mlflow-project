import sys
import os
import pytest
import pandas as pd

# Ajouter le répertoire racine au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train_model import train_model

def test_train_model():
    # Vérifier que les données existent
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.csv')
    data = pd.read_csv(data_path)
    assert data.shape[0] > 0
    assert data.shape[1] == 4
    
    # Vérifier que l'entraînement fonctionne
    model, mse, r2 = train_model()
    
    assert model is not None
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert mse >= 0