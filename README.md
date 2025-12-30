
# TMDb — Pipeline ElasticNet (TF‑IDF + SVD + Numériques + CV)

Ce dépôt contient un script complet de modélisation pour prédire le revenu des films à partir du dataset **TMDb 5000**. 
Le pipeline inclut : parsing JSON (genres, keywords, studios, pays, langues, cast/crew, title), TF‑IDF par blocs avec compression SVD, 
transformations numériques (Yeo‑Johnson, log1p), utilisation du modèle ElasticNet qui a donné de meilleurs résultats par rapport au Random Forest ou CatBoost avec TransformedTargetRegressor pour reconvertir la prédiction dans la valeur de la cible initiale, RandomizedSearchCV pour optimiser les hyperparamètres, et un rapport d'évaluation des valeurs RMSE/MAE/R²/SMAPE + figures résidus, QQ‑plot, réel vs prédit.


## Structure du dépôt
```text
tmdb-elasticnet-pipeline/
├── src/
│   └── main.py                # Script principal (le code fourni)
├── notebooks/                 # EDA, analyses complémentaires
├── outputs/                   # Prédictions, métriques, figures
├── data/                      # (optionnel) données locales
├── configs/                   # (optionnel) fichiers de configuration
├── tests/                     # (optionnel) tests unitaires
├── docs/                      # Documentation additionnelle
├── requirements.txt           # Dépendances Python
├── .gitignore                 # Ignore cache/venv/data/outputs
└── README.md                  # Ce fichier
```



## Données
- **Source** : Kaggle — *TMDb 5000 Movie Dataset* via `kagglehub` (`tmdb/tmdb-movie-metadata`).
- Le script télécharge automatiquement les fichiers `tmdb_5000_movies.csv` et `tmdb_5000_credits.csv`.

## Prérequis
- Python 3.10+
- Internet (pour `kagglehub`)
- `pip`

## Installation
```bash
git clone <URL_DU_DEPOT_GITHUB>.git
cd tmdb-elasticnet-pipeline

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution
```bash
python src/main.py
```

Le script :
- Télécharge les données via `kagglehub`
- Prépare les colonnes (dates → year/month/day_of_week, parsing JSON, transformations)
- Construit TF‑IDF par blocs + SVD
- Monte un `ColumnTransformer` (TF‑IDF compressé + numériques)
- Entraîne **ElasticNet** via `TransformedTargetRegressor` (cible Yeo‑Johnson)
- Cherche les hyperparamètres (`RandomizedSearchCV`) sur `KFold` stratifié par quantiles de la cible
- Évalue sur le test et génère plots + fichiers de sortie

## 📈 Sorties
- Prédictions test : `outputs/tmdb_elasticnet_predictions_test.csv`
- Corrélations (train) :
  - `outputs/tmdb_correlations_train_yj.csv`
  - `outputs/tmdb_correlations_train_raw.csv`
- Figures d'évaluation (affichées à l'écran; vous pouvez les sauvegarder dans `outputs/figures/` si vous le souhaitez)

## 🧪 Qualité
- Code commenté et structuré.
- Reproductibilité : `RANDOM_STATE` fixé, versions figées.

## 📝 Licence
- MIT — voir `LICENSE`.

## 👤 Auteur
- CABANA David — Digital Monitoring Center Coordinator
