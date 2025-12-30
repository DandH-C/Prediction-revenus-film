
# TMDb — Prédiction des revenus des films

Projet de modélisation pour prédire le revenu mondial des films à partir des métadonnées TMDb.
Le dépôt propose un pipeline complet (prétraitement JSON → TF‑IDF → SVD → features numériques →
modèles ElasticNet/Tweedie) et une évaluation sur un jeu de test tenu à part.

## 🎯 Objectifs
- Produit final fonctionnel (exécutable de bout en bout)
- Organisation, complétude, pertinence, efficience et qualité

## 🗂️ Structure
```text
tmdb-revenue-prediction/
├── src/
│   ├── main.py                # Script principal A/B (ElasticNet vs Tweedie)
│   ├── preprocessing.py        # Fonctions de parsing JSON & features
│   ├── modeling.py             # Construction des pipelines & CV
│   └── utils.py                # Métriques, I/O, logs
├── notebooks/                  # EDA & essais
├── outputs/                    # Prédictions, métriques, figures
├── data/                       # (optionnel) données locales / README
├── configs/                    # Fichiers de configuration YAML/JSON
├── tests/                      # Tests unitaires
├── docs/                       # Docs additionnelles
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 🚀 Installation
```bash
git clone <URL_DU_DEPOT_GITHUB>.git
cd tmdb-revenue-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Exécution (Version A/B)
```bash
# Variante A (ElasticNet + Yeo-Johnson, CV temporelle)
python src/main.py --version A --cv time --transform yeo-johnson --n-folds 6 --n-iter 60

# Variante B (Tweedie, CV temporelle)
python src/main.py --version B --cv time --n-folds 6 --n-iter 60
```

Paramètres : `--version [A|B]`, `--transform [yeo-johnson|log1p|none]`, `--cv [time|group|strat_kfold]`, `--clip-negative`.

## 📈 Sorties
- `outputs/tmdb_predictions_test_<VERSION>.csv`
- `outputs/metrics_<VERSION>.json`
- `outputs/figures/`

## 🔍 Reproductibilité
- `RANDOM_STATE` fixé dans le code
- versions figées dans `requirements.txt`
- CV temporelle

## 📝 Licence
MIT — voir `LICENSE`

## 👤 Auteur
CABANA David
