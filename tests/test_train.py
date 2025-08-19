import pytest
import pandas as pd
from models.train_model import train_model

def test_train_model():
    # Vérifier que les données existent
    data = pd.read_csv('../data/sample_data.csv')
    assert data.shape[0] > 0
    assert data.shape[1] == 4
    
    # Vérifier que l'entraînement fonctionne
    model, mse, r2 = train_model()
    
    assert model is not None
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert mse >= 0