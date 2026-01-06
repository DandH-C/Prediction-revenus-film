# Explications des résultats

## Mise en contexte
L'objectif du projet est d'estimer les revenus d'un film en fonction des ses métadonnées. La source utilisées provient de TMDb et porte sur un ensemble d'environ 4800 films. Chaque films contient les colonnes suivantes: Budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average et vote_count. Pour simplifier le modèle, je n'ai pas considéré ces colonnes: id, movie_id, homepage, original_language, status, tagline, overview, original_title, vote_count. De plus, puisqu'il m'était impossible de déterminer si un budget ou un revenu de 0 était vraiment de 0 et non pas manquant, j'ai supprimé ces lignes. À la fin du nettoyage j'avais 3376 lignes. 

## Hypothèse initiales
L'hypothèse alternative initiale voulait qu'au moins un des attributs de la base de données ait un impact significatif sur les revenus générés. Elle est confirmée par ce tableau: <img width="639" height="146" alt="image" src="https://github.com/user-attachments/assets/281cff92-01ff-4cc6-bdb3-d9c24ea60ea2" /> On constate que popularity_log et budget_yj ont un impact car leurs valeurs P est < 0,05. Nous pouvons rejetter l'hypothèse nulle.

## Résultats obtenus


## Score final et conclusion
