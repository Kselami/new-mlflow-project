# Script PowerShell pour exécuter le pipeline complet
Write-Host "Démarrage du pipeline MLOps..." -ForegroundColor Green

# Activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1

# Exécuter les tests
Write-Host "Exécution des tests..." -ForegroundColor Yellow
pytest tests/ -v

# Entraîner le modèle
Write-Host "Entraînement du modèle..." -ForegroundColor Yellow
python models/train_model.py

# Vérifier que les métriques ont été générées
if (Test-Path "mlflow_output/metrics.txt") {
    Write-Host "Métriques enregistrées:" -ForegroundColor Green
    Get-Content "mlflow_output/metrics.txt"
} else {
    Write-Host "Erreur: les métriques n'ont pas été générées" -ForegroundColor Red
    exit 1
}

Write-Host "Pipeline exécuté avec succès!" -ForegroundColor Green