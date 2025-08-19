import pytest
import numpy as np
from models.predict import make_prediction

def test_make_prediction():
    # Créer un modèle mock pour tester la prédiction
    class MockModel:
        def predict(self, features):
            return np.array([15.0])
    
    model = MockModel()
    sample_features = np.array([[2.5, 4.8, 6.1]])
    
    prediction = make_prediction(model, sample_features)
    assert prediction[0] == 15.0