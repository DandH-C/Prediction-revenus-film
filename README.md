# Prediction des revenus d'un film - Analyse exploratoire et apprentissage machine avec ElasticNet

# Objectifs:
Création d'un modèle prédictif pour estimer les revenus d'un film en fonction des métadonnées publiques TMDB.
Le projet ci-joint fournit le code complet et reproductible basé sur scikit-learn

Étapes inclues dans ce projet
- Importation des données;
- Traitement de la variable cible (Yeo-Johnson) et nettoyage des données;
- Analyse syntaxique JSON sécurisé des colonnes textes;
- Application TF-IDF avec Top-N suivi de SVD pour compression;
- Utilisation de Yeo-Johnson sur budget et vote_average, log1p sur popularity
- Modèle ElasticNet optimisé avec RandomSearchCV
- Application de TransformedTargetRegressor sur la cible pour obtenir les métriques en dollars
- Métriques: RMSE / MAE / R2 / SMAPE et graphiques des résidus, QQ-Plot
- Export csv des prédictions
Instructions
