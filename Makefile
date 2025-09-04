SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Créer le dossier data
data:
	mkdir data

# Charger les données brutes
data/raw_dataset.csv: data
	python -m src.load_data

# Nettoyer les données  
data/clean_dataset.csv: data/raw_dataset.csv
	python -m src.clean_data

# Préprocesser les données (crée 4 fichiers en une fois)
data/X_train.csv: data/clean_dataset.csv
	python -m src.preprocess_data

# Entraîner les modèles et sélectionner le meilleur
data/model.pkl: data/X_train.csv
	python -m src.training

# Évaluer le modèle final
data/evaluation_metrics.csv: data/model.pkl
	python -m src.evaluate

# Pipeline complet
pipeline: data/evaluation_metrics.csv

# Voir les résultats de sélection des modèles
results:
	python -c "import pandas as pd; print(pd.read_csv('data/model_selection_results.csv'))"

# Voir les métriques finales
metrics:
	python -c "import pandas as pd; print(pd.read_csv('data/evaluation_metrics.csv').T)"

# Installer les dépendances
install:
	pip install -r requirements.txt

# Nettoyer tous les fichiers générés
clean:
	rm -rf data/

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make install   - Installer les dépendances"
	@echo "  make pipeline  - Exécuter le pipeline complet"
	@echo "  make results   - Voir la comparaison des modèles"
	@echo "  make metrics   - Voir les métriques d'évaluation"
	@echo "  make clean     - Supprimer les fichiers générés"
	@echo "  make help      - Afficher cette aide"

.PHONY: pipeline results metrics install clean help