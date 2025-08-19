# Script PowerShell pour configurer l'environnement
Write-Host "Configuration de l'environnement MLOps..." -ForegroundColor Green

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python -c "import mlflow; print('MLflow version:', mlflow.__version__)"

Write-Host "Environnement configuré avec succès!" -ForegroundColor Green