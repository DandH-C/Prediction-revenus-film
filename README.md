
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
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

## Données
- **Source** : Kaggle ([tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)) — *TMDb 5000 Movie Dataset* via l'api kagglehub.

## Prérequis
- Python 3.10+
- Installation des librairies contenues dans le fichier requirements.txt

## Exécution locale
```bash
git clone <URL_DU_DEPOT_GITHUB>.git
cd tmdb-elasticnet-pipeline
```
### Lancement de l'environnement virtuel
```bash
python -m venv .venv

.venv\Scripts\activate
```
### Installation des requis
```bash
pip install -r requirements.txt
```
### Récupération des fichiers csv sauvegardés localement
Pour spécifier le chemin des fichiers sauvegardés en local, utilisez cette syntaxe:
```bash
'python main.py --data-source local --data-dir ./<mondossier>'
```
Assurez-vous que le dossier spécifié contienne les deux fichiers .csv.

## Exécution en ligne
Copier/coller le script fourni dans un notebook Colab ou Jupyter.
Le script télécharge automatiquement les fichiers `tmdb_5000_movies.csv` et `tmdb_5000_credits.csv`.

## Le script :
- Prépare les colonnes (dates → year/month/day_of_week, parsing JSON, transformations).
- Construit TF‑IDF par blocs + SVD.
- Monte un ColumnTransformer (TF‑IDF compressé + numériques).
- Entraîne ElasticNet via TransformedTargetRegressor (cible Yeo‑Johnson).
- Cherche les hyperparamètres RandomizedSearchCV sur KFold stratifié par quantiles de la cible.
- Évalue sur le test et génère plots + fichiers de sortie.

## Sorties
- Prédictions test : `outputs/predictions_test.csv`
- Corrélations (train) :
  - `outputs/tmdb_correlations_train_yj.csv`
  - `outputs/tmdb_correlations_train_raw.csv`
- Figures d'évaluation: affichées à l'écran; vous pouvez les sauvegarder si vous le souhaitez.

## Licence
Ce projet est sous licence **MIT**.

Vous êtes libre d'utiliser, copier, modifier, fusionner, publier, distribuer, sous-licencier et/ou vendre des copies du logiciel, sous réserve d'inclure la mention de copyright et la présente notice dans toutes les copies ou parties substantielles du logiciel.

Consultez le fichier [LICENSE](LICENSE) pour plus de détails

## Auteur
David Cabana
