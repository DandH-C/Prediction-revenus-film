
# TMDb — Pipeline ElasticNet (TF‑IDF, SVD et CV)

Ce dépôt contient un script complet de modélisation pour prédire le revenu des films à partir du dataset **TMDb 5000**. 
Le pipeline inclut : parsing JSON (genres, keywords, studios, pays, langues, cast/crew, title), TF‑IDF par blocs avec compression SVD, 
transformations numériques (Yeo‑Johnson, log1p), utilisation du modèle ElasticNet qui a donné de meilleurs résultats par rapport au Random Forest ou CatBoost avec TransformedTargetRegressor pour reconvertir la prédiction dans la valeur de la cible initiale, RandomizedSearchCV pour optimiser les hyperparamètres, et un rapport d'évaluation des valeurs RMSE/MAE/R²/SMAPE + figures résidus, QQ‑plot, réel vs prédit.


## Structure du dépôt
```text
tmdb-elasticnet-pipeline/
├── src/
│   └── main.py                # Script principal
├── outputs/                   # Prédictions, métriques, figures
├── requirements.txt           # Dépendances Python
├── resultats                  # Résultats et explications
└── README.md                  # Ce fichier
```

## Données
- **Source** : Kaggle — *TMDb 5000 Movie Dataset*.

## Prérequis
- Python 3.10+
- pip 25+

## Exécution locale
- Téléchargez les fichiers csv qui se trouvent sous ici:([tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)).
- Cliquez télécharger le zip pour ce GitHub et décompresser-le sur votre machine.
- Installez les librairies nécessaire:
```bash
pip install -r requirements.txt
```
  
### Lancement de l'environnement virtuel
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Exécution et récupération des fichiers csv sauvegardés localement
Pour spécifier le chemin des fichiers sauvegardés en local, utilisez cette syntaxe:
```bash
python main.py --data-source local --data-dir ./<mondossier>
```
Remplacez 'mondossier' par votre nom de dossier actuel. Assurez-vous que celui-ci contienne les deux fichiers .csv.

## Exécution en ligne
Copier/coller le script fourni dans un notebook Colab ou Jupyter.
Le script télécharge automatiquement les fichiers `tmdb_5000_movies.csv` et `tmdb_5000_credits.csv`.

## Le script :
- Prépare les colonnes (dates -> year/month/day_of_week, parsing JSON, transformations).
- Construit TF‑IDF par blocs + SVD.
- Monte un ColumnTransformer (TF‑IDF compressé + numériques).
- Entraîne ElasticNet via TransformedTargetRegressor et la cible transformée Yeo‑Johnson.
- Cherche les hyperparamètres RandomizedSearchCV sur KFold stratifié par quantiles de la cible.
- Évalue sur le test et génère plots + fichiers de sortie.
- Certaines portions sont commentées. Elles ont été utiles lors de la création du script pour mieux comprendre les données.

## Sorties
- Prédictions: `outputs/predictions_test.csv`
- Corrélations (train):
  - `outputs/tmdb_correlations_train_yj.csv`
  - `outputs/tmdb_correlations_train_raw.csv`
- Figures d'évaluation: affichées à l'écran; vous pouvez les sauvegarder si vous le souhaitez.

## Licence
Ce projet est sous licence **MIT**.

Vous êtes libre d'utiliser, copier, modifier, fusionner, publier, distribuer, sous-licencier et/ou vendre des copies du logiciel, sous réserve d'inclure la mention de copyright et la présente notice dans toutes les copies ou parties substantielles du logiciel.

Consultez le fichier [LICENSE](LICENSE) pour plus de détails

## Auteur
David Cabana
