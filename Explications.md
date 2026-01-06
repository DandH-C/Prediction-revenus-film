# Explications des résultats
## Mise en contexte
L’objectif du projet est d’estimer les revenus d’un film en fonction de ses métadonnées. La source utilisée provient de TMDb et porte sur un ensemble d’environ 4 800 films. Chaque film contient les colonnes suivantes : budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average et vote_count.

Pour simplifier le modèle, je n’ai pas conservé les colonnes suivantes : id, movie_id, homepage, original_language, status, tagline, overview, original_title, vote_count.
De plus, puisqu’il était impossible de déterminer si un budget ou un revenu égal à 0 correspondait réellement à zéro ou à une valeur manquante, j’ai supprimé ces lignes. Après nettoyage, il restait 3 376 lignes.

## Hypothèse initiale
L’hypothèse alternative initiale stipulait qu’au moins un des attributs de la base de données avait un impact significatif sur les revenus générés. 
Elle est confirmée par ce tableau :
<img width="637" height="138" alt="image" src="https://github.com/user-attachments/assets/5d3d889c-53f6-4751-b57b-f8c6e7fd628d" />
On constate que popularity_log et budget_yj ont un impact, car leurs valeurs de p sont inférieures à 0,05. Nous pouvons donc rejeter l’hypothèse nulle.

## Résultats obtenus
Les résultats ne sont pas à la hauteur de mes attentes. Un facteur majeur concerne le nombre de lignes dans mon jeu de données. Initialement, il était acceptable, mais l’exclusion des valeurs aberrantes ainsi que des valeurs nulles ou égales à zéro a contribué à le réduire considérablement.
Un autre facteur est la période couverte par les données : les films s’échelonnent entre 1920 et 2015. Les budgets consacrés aux films il y a près de 100 ans n’ont rien à voir avec les montants actuels, ce qui introduit une forte hétérogénéité.
