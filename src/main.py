
    # Imports
    import warnings
    warnings.filterwarnings("ignore")

    import os, json, re, unicodedata, math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as st
    from scipy.stats import skew, pearsonr, spearmanr
    from matplotlib.ticker import ScalarFormatter
    from IPython.display import display

    import kagglehub
    import statsmodels.api as sm

    from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler, PolynomialFeatures
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Paramètres
    RANDOM_STATE = 42
    TARGET_COL = "revenue"

    JSON_COLS = [
        "genres", "keywords", "production_companies", "production_countries",
        "spoken_languages", "cast", "crew", "title"
    ]

    # Valeurs min/max pour la sélection des mots
    MIN_DF, MAX_DF = 6, 0.85
    TOP_N_CAST, TOP_N_CREW = 180, 180
    TITLE_MAX_FEATURES = 5000

    # Compression SVD pour éviter les blocs dense
    SVD_PER_BLOCK = {"cast":160, "crew":160, "kw":400, "genres":20, "pc":300, "cty":40, "lang":40, "title":120}

    # Paramètres K-fold
    N_FOLDS = 6    # Pour une exécution rapide mais moins précise, choisir entre 3 et 6
    N_ITER  = 150  # Ici, entre 40 et 300

    # =========================================================
    # 1) Chargement des données sous KaggleHub
    # =========================================================
    data_folder = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print("Téléchargement KaggleHub ->", data_folder)

    df_movies  = pd.read_csv(os.path.join(data_folder, "tmdb_5000_movies.csv"))
    df_credits = pd.read_csv(os.path.join(data_folder, "tmdb_5000_credits.csv"))

    # =========================================================
    # 2) Préparation / Filtrage
    # =========================================================
    # Imposer type numérique
    for c in ["budget", "revenue", "runtime"]:
        if c in df_movies.columns:
            df_movies[c] = pd.to_numeric(df_movies[c], errors="coerce")

    # Vérifier colonnes title si elles sont identiques
    col1 = df_movies["title"]; col2 = df_credits["title"]
    print("
Les 2 colonnes title sont identiques?", col1.equals(col2))

    # Drop 'title' côté crédits
    df_credits = df_credits.drop(columns=["title"], errors="ignore")

    # Fusion les 2 datasets
    df = df_movies.merge(df_credits, left_on="id", right_on="movie_id", how="inner")

    # Colonnes à supprimer pour simplifier le modèle
    drop_cols = ["id", "movie_id", "homepage", "original_language", "status", "tagline", "overview", "original_title", "vote_count"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Masque pour budget <=0 ou revenue <=0
    bud_rev_0 = (df["budget"] <= 0) | (df["revenue"] <= 0)

    print("Avant filtre: budget <= 0 OU revenue <= 0:", int(bud_rev_0.sum()), "lignes sur", len(df))
    df = df.loc[~bud_rev_0].copy()

    print("
Après filtre, les variables revenue et budget:")
    print("% revenue == 0:", (df["revenue"] == 0).mean() * 100, "| % budget == 0:", (df["budget"] == 0).mean() * 100)

    # Date -> calendrier
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month
    df["day_of_week"] = df["release_date"].dt.dayofweek
    df = df.drop(columns=["release_date"])

    # =========================================================
    # 3) Variable cible + EDA initiale
    # =========================================================
    y_raw = pd.to_numeric(df["revenue"], errors="coerce")
    mask = y_raw.notna()
    y = y_raw[mask].to_numpy().reshape(-1, 1)

    # Visualiser l'asymétrie de la variable brut
    print(f"Asymétrie avant: {y_raw.skew():.3f}")

    # Histogramme de l'asymétrie de la variable brut
    plt.figure(figsize=(8, 4))
    plt.hist(y_raw.dropna(), bins=50, color="steelblue", alpha=0.8)
    plt.title("Distribution des revenus avant transformation")
    plt.xlabel("Revenue"); plt.ylabel("Fréquences")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    plt.ticklabel_format(style="plain", axis="y")
    plt.show()

    # Transformation Yeo-Johnson
    y_yj = PowerTransformer(method="yeo-johnson").fit_transform(y)
    print(f"Asymétrie après Yeo-Johnson: {skew(y_yj.flatten(), bias=False):.3f}")

    # Ajout colonne revenue_yj
    df["revenue_yj"] = np.nan
    df.loc[mask, "revenue_yj"] = y_yj.flatten()

    # Histogramme de l'asymétrie de la variable transformée YJ
    plt.figure(figsize=(8, 4))
    plt.hist(y_yj, bins=50, color="darkorange", alpha=0.8)
    plt.title("Distribution des revenus après transformation")
    plt.xlabel("revenus_YJ"); plt.ylabel("Fréquences")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    plt.ticklabel_format(style="plain", axis="y")
    plt.show()

    # Visualiser si la nouvelle variable suit une distribution normale
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    st.probplot(df["revenue"].dropna(), dist="norm", plot=ax[0]); ax[0].set_title("QQ-plot: revenue brut")
    st.probplot(df["revenue_yj"].dropna(), dist="norm", plot=ax[1]); ax[1].set_title("QQ-plot: revenue YJ")
    plt.show()

    # Shape du dataframe
    print(f"Shape finale: {df.shape}")

    # Voir si nous avons des valeurs absentes
    print("
Valeurs nulles ou NaN par colonne pour df:")
    display(df.isna().sum())  # nombre de NaN par colonne

    # Afficher le nom des colonnes
    print("
Colonnes finales:", df.columns.tolist())

    # =========================================================
    # 4) Parsing JSON (normalisation des textes + tokens)
    # =========================================================
    # Normalisation du texte, retrait des accents, tout en minuscule etc.
    def normalize_text(text: str, keep_phrase: bool = False) -> str:
        if not isinstance(text, str): text = str(text or "")
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").lower().strip()
        text = " ".join(text.split())
        if keep_phrase:
            text = re.sub(r"[^a-z0-9_]+", " ", text)
            text = "_".join(text.split())
        return text

    # Si le texte json contient un dictionnaire, l'extraire
    def extract_dict(d: dict, keep_phrase: bool) -> list:
        names = []
        if isinstance(d, dict) and "name" in d:
            nt = normalize_text(d["name"], keep_phrase=keep_phrase)
            if nt: names.append(nt)
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, dict):
                    names.extend(extract_dict(v, keep_phrase))
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            names.extend(extract_dict(it, keep_phrase))
        return names

    # Application des token soit 'aucun' lorsque la valeur est absente ou ';' lorsque plusieurs mots à séparer
    def json_conv(cell, empty_token: str = "aucun", sep: str = ";", keep_phrase: bool = False) -> str:
        if pd.isna(cell): return empty_token
        if isinstance(cell, str):
            s = cell.strip()
            if not s: return empty_token
            try:
                cell = json.loads(s)
            except json.JSONDecodeError:
                return normalize_text(s, keep_phrase=keep_phrase) or empty_token
        names = []
        if isinstance(cell, list):
            for item in cell:
                if isinstance(item, dict):
                    names.extend(extract_dict(item, keep_phrase))
                elif isinstance(item, str):
                    stxt = item.strip()
                    if stxt:
                        try:
                            parsed = json.loads(stxt)
                            if isinstance(parsed, dict):
                                names.extend(extract_dict(parsed, keep_phrase))
                        except json.JSONDecodeError:
                            pass
        return sep.join(names) if names else empty_token

    # Copie pour analyse et parsing JSON
    df_json_parse = df.copy()
    for col in JSON_COLS:
        if col in df_json_parse.columns:
            empty_token = "inconnu" if col == "spoken_languages" else "aucun"
            keep_phrase = col in {"genres", "keywords", "production_companies", "production_countries", "cast", "crew"}
            df_json_parse[col] = df_json_parse[col].apply(lambda x: json_conv(x, empty_token=empty_token, sep=";", keep_phrase=keep_phrase))
            if col == "keywords":
                df_json_parse[col] = df_json_parse[col].apply(
                    lambda s: (";".join(sorted(set([t for t in s.strip().split(";") if t]))))
                      if isinstance(s, str) and s.strip()
                      else "aucun"
                )

    # Vérification problèmes JSON
    json_vide = [{"Colonne": col, "NaN": df_json_parse[col].isna().sum(), "Vides": (df_json_parse[col].str.strip() == "").sum()}
        for col in JSON_COLS
                 if col in df_json_parse.columns]

    table_probleme = pd.DataFrame(json_vide).sort_values(by="NaN", ascending=False)
    print("Problèmes potentiels:")
    print(table_probleme)

    print("
Affichage des données parsé json")
    print("
", df_json_parse.head())

    # =========================================================
    # 5) Analyse sur les variables numériques
    # =========================================================
    num_col = ["budget", "popularity", "runtime", "vote_average", "year", "month", "day_of_week"]
    print(df_json_parse[num_col].describe())

    # Catégoriser revenue_yj en quantiles pour boxplots
    df_json_parse["revenue_category"] = pd.qcut(df_json_parse["revenue_yj"], q=4, duplicates="drop")
    df_json_parse["revenue_category"] = df_json_parse["revenue_category"].apply(lambda x: pd.Interval(round(x.left, 2), round(x.right, 2)))

    a = max(1, math.floor(df_json_parse.shape[1] / 3))

    # Affichage des boxplots en fonction des attributs
    plt.figure(figsize=(20, 30))
    for i, col in enumerate(num_col, 1):
        plt.subplot(a, 3, i)
        sns.boxplot(x="revenue_category", y=col, hue="revenue_category", data=df_json_parse, palette="Set2", legend=False)
        plt.title(f"{col} vs revenue Yeo Johnson")
        plt.xlabel("Revenue Categorie"); plt.ylabel(col)
    plt.tight_layout(); plt.show()

    # Création des asymétrie des colonnes numériques originales
    sk_num = df_json_parse[num_col].skew(numeric_only=True)
    print("Asymetrie des variables numériques:")
    print(sk_num.sort_values(ascending=False))

    # =========================================================
    # 6) Transformations numériques en Yeo-Johnson ou log1p
    # =========================================================
    # Transformation pour popularity, budget et vote_average
    df_json_parse["popularity_log"] = np.log1p(df_json_parse["popularity"])

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("power", PowerTransformer(method="yeo-johnson", standardize=False)), ("scaler", StandardScaler())])

    df_json_parse["budget_yj"] = num_pipe.fit_transform(df_json_parse[["budget"]])
    df_json_parse["vote_average_yj"] = num_pipe.fit_transform(df_json_parse[["vote_average"]])

    # Drop des originaux
    df_json_parse = df_json_parse.drop(columns=["budget", "vote_average", "popularity"])

    new_num_col = ["budget_yj", "popularity_log", "vote_average_yj"]

    # Boxplot des relations entre les attributs et la variable cible transformée
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(new_num_col, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x="revenue_category", y=col, hue="revenue_category", data=df_json_parse, palette="Set2", legend=False)
        plt.title(f"{col} vs revenue Yeo Johnson")
        plt.xlabel("Revenu Categorie"); plt.ylabel(col)
    plt.tight_layout(); plt.show()

    # Création des asymétrie des colonnes numériques transformées
    print("Asymétrie des variables numériques transformées:")
    print(df_json_parse[new_num_col].skew(numeric_only=True).sort_values(ascending=False))

    # =========================================================
    # 7) Modélisation
    # =========================================================
    # Définition des jeux d'entraînement et de test
    Y = df_json_parse[TARGET_COL].astype(float)
    X = df_json_parse.drop(columns=[TARGET_COL])

    # 20% test et 80% entrainement
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=RANDOM_STATE)
    print("Split:", X_train.shape, X_test.shape)

    # Application de TF-IDF pour déterminer l'importance des mots
    def tfidf_fit_transform(s_train: pd.Series, s_test: pd.Series, as_title=False):
        if as_title:
            vect = TfidfVectorizer(
                lowercase=True, stop_words="english",
                ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=TITLE_MAX_FEATURES
            )
        else:
            vect = TfidfVectorizer(
                lowercase=False, token_pattern=None,
                tokenizer=lambda s: [t.strip() for t in s.split(";") if t.strip()],
                min_df=MIN_DF, max_df=MAX_DF, ngram_range=(1, 1)
            )
        Xtr = vect.fit_transform(s_train.fillna("").astype(str))
        Xte = vect.transform(s_test.fillna("").astype(str))
        return Xtr, Xte

    # TF-IDF par bloc JSON
    tfidf_tr_parts, tfidf_te_parts = [], []
    for col in JSON_COLS:
        if col not in X_train.columns: continue
        as_title = (col == "title")
        Xtr_block, Xte_block = tfidf_fit_transform(X_train[col], X_test[col], as_title=as_title)

        # Sélection des Top-N pour cast/crew pour limiter la variabilité
        top_n = TOP_N_CAST if col == "cast" else (TOP_N_CREW if col == "crew" else None)
        if top_n:
            sums = np.asarray(Xtr_block.sum(axis=0)).ravel()
            order = np.argsort(-sums)
            keep_idx = order[:min(top_n, Xtr_block.shape[1])]
            Xtr_block = Xtr_block[:, keep_idx]; Xte_block = Xte_block[:, keep_idx]

        # SVD (compression) par bloc json
        block_name = {"keywords":"kw","production_companies":"pc","production_countries":"cty","spoken_languages":"lang"}.get(col, col)
        n_feat = Xtr_block.shape[1]
        n_comp = max(1, min(SVD_PER_BLOCK.get(block_name, 150), n_feat - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        Ztr = svd.fit_transform(Xtr_block).astype(np.float32)
        Zte = svd.transform(Xte_block).astype(np.float32)
        cols = [f"{block_name}_svd_{i+1}" for i in range(n_comp)]
        tfidf_tr_parts.append(pd.DataFrame(Ztr, index=X_train.index, columns=cols))
        tfidf_te_parts.append(pd.DataFrame(Zte, index=X_test.index, columns=cols))

    Xtr_tfidf = pd.concat(tfidf_tr_parts, axis=1) if tfidf_tr_parts else pd.DataFrame(index=X_train.index)
    Xte_tfidf = pd.concat(tfidf_te_parts, axis=1) if tfidf_te_parts else pd.DataFrame(index=X_test.index)

    print(f"
TF-IDF Train: {Xtr_tfidf.shape} | Test: {Xte_tfidf.shape}")

    # Attributs numérique, transformation YJ pour train et test
    def transform_numeric_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame):
        tr, te = df_train.copy(), df_test.copy()

        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        num_tr = pd.DataFrame({
            "budget": pd.to_numeric(tr.get("budget_yj", tr.get("budget", 0.0)), errors="coerce"),
            "vote_average": pd.to_numeric(tr.get("vote_average_yj", tr.get("vote_average", 0.0)), errors="coerce")
        }).fillna(0.0)
        num_te = pd.DataFrame({
            "budget": pd.to_numeric(te.get("budget_yj", te.get("budget", 0.0)), errors="coerce"),
            "vote_average": pd.to_numeric(te.get("vote_average_yj", te.get("vote_average", 0.0)), errors="coerce")
        }).fillna(0.0)

        # Les colonnes transformées sont insérées dans tr et te
        yj_tr = pt.fit_transform(num_tr); yj_te = pt.transform(num_te)
        tr["budget_yj"], tr["vote_average_yj"] = yj_tr[:,0], yj_tr[:,1]
        te["budget_yj"], te["vote_average_yj"] = yj_te[:,0], yj_te[:,1]

        # Colonnes qui doivent être retournées par la fonction
        keep = ["budget_yj", "popularity_log", "runtime", "vote_average_yj", "year", "month", "day_of_week"]
        tr = tr[keep].fillna(0.0); te = te[keep].fillna(0.0)

        return tr, te

    Xtr_num, Xte_num = transform_numeric_train_test(X_train.copy(), X_test.copy())

    # Évaluation des corrélations entre attributs et variable cible
    def correlations_with_target(df: pd.DataFrame, feat_cols: list, target_col: str = "revenue_yj"):
        rows = []
        for col in feat_cols:
            if col not in df.columns:
                continue
            # Forcer le type numérique
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[target_col], errors="coerce")

            # Masque pour les lignes valides seulement
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                rows.append({"feature": col, "pearson_r": np.nan, "pearson_p": np.nan,
                             "spearman_rho": np.nan, "spearman_p": np.nan, "n": int(valid.sum())})
                continue

            # Calcul des corrélations + p-values (linéaires et monotone)
            r,  p  = pearsonr(x[valid], y[valid])
            rs, ps = spearmanr(x[valid], y[valid])
            rows.append({"feature": col, "pearson_r": r, "pearson_p": p,
                         "spearman_rho": rs, "spearman_p": ps, "n": int(valid.sum())})

        # Assemblage du df
        out = pd.DataFrame(rows)
        if not out.empty:
            out["abs_pearson"] = out["pearson_r"].abs()
            out = out.sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"])
        # Tableau contenant les colonnes features, pearson_r, pearson_p, spearman_rho, spearman_p et n.
        return out.reset_index(drop=True)

    # Assemblage des df train et test
    Xtr_all = pd.concat([Xtr_tfidf, Xtr_num], axis=1)
    Xte_all = pd.concat([Xte_tfidf, Xte_num], axis=1)
    print("Shapes assemblées (compressées):", Xtr_all.shape, Xte_all.shape)

    # Liste des colonnes TF-IDF
    tfidf_cols = [c for c in Xtr_all.columns if any(c.startswith(p) for p in ("cast_", "crew_", "genres_", "kw_", "pc_", "cty_", "lang_", "title_"))]

    # Liste des colonnes numériques modifiées
    core_num   = ["budget_yj","popularity_log","runtime","vote_average_yj","year","month","day_of_week"]

    # Filtre des noms de colonnes valides
    core_num   = [c for c in core_num if c in Xtr_all.columns]

    # Évaluer si des synergies existent entre les variables numériques et mise à l'échelle
    core_pipe = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)), ("scaler", RobustScaler())])

    # Traitement en fonction du groupe
    prep = ColumnTransformer(transformers=[("tfidf_comp", StandardScaler(with_mean = False), tfidf_cols), ("num_core", core_pipe, core_num)],remainder="drop")

    # Modèle ElasticNet + cible YJ avec TTR
    base = Pipeline([("preprocessor", prep), ("clf", ElasticNet(random_state=RANDOM_STATE, max_iter=10000))])

    # Réduction de l'asymétrie avec YJ lors de l'entraînement et retour en échelle normale pour les revenus
    ttr = TransformedTargetRegressor(regressor=base, transformer=PowerTransformer(method="yeo-johnson", standardize=True))

    # Hyperparamètres du modèle
    param_dist = {
        "regressor__clf__alpha": np.unique(np.concatenate([
                  np.logspace(-4.5, -0.5, 30),
                  np.linspace(0.005, 0.12, 30)])),

        "regressor__clf__l1_ratio": np.linspace(0.05, 0.9, 12)}

    # CV par quantiles sur y_train avec K-Fold
    q = pd.qcut(y_train.rank(method="first"), q=N_FOLDS, labels=False, duplicates="drop")
    cv_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y_train)), q))

    ############################
    ##### To be completed #####
    search = RandomizedSearchCV(
        estimator=ttr,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1)

    # Recherche des meilleurs paramètres
    search.fit(Xtr_all, y_train)
    cv_rmse = -search.best_score_
    print("Best params:", search.best_params_)
    print(f"Best CV RMSE: {cv_rmse:.2f}")

    # Définition de l'erreur symétrique entre y_true et y_pred (réduction des erreurs avec des valeurs près de 0)
    # Obtenir la plus petite valeur possible
    def smape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
        denom = (np.abs(y_true) + np.abs(y_pred)); denom[denom == 0] = 1.0
        return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

    # Début des prédictions et RMSE dans la même unité que la cible
    def evaluate(estimator, Xte, yte, ytr, title="Test"):
        y_pred = estimator.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, y_pred))
        # Génération du mae pour les outliers
        mae  = mean_absolute_error(yte, y_pred)
        r2   = r2_score(yte, y_pred)
        # Comparer avec une prédiction faisant la moyenne seulement
        ybar = np.full_like(yte, fill_value=ytr.mean())
        rmse_base = np.sqrt(mean_squared_error(yte, ybar))
        gain = 100.0 * (1.0 - rmse / rmse_base)

        # Pourcentage d'erreur sur les montants initiaux
        s_mape = smape(yte, y_pred)

        print(f"
=== Évaluation {title} ===")
        print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | SMAPE: {s_mape:.2f}%")
        print(f"Baseline RMSE: {rmse_base:.2f} | Gain vs baseline: {gain:.1f}%")
        return {"rmse": rmse, "mae": mae, "r2": r2, "smape": s_mape, "rmse_base": rmse_base, "gain_pct": gain}

    # Exécution des métriques pour trouver le meilleur ajustement possible
    test_metrics = evaluate(search.best_estimator_, Xte_all, y_test, y_train, title="Test")

    # Plots résidus
    y_pred = search.best_estimator_.predict(Xte_all)
    residuals = y_test - y_pred

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plt.hist(residuals, bins=40, color="steelblue", alpha=0.85)
    plt.title("Histogramme des résidus"); plt.xlabel("Résidu ($)"); plt.ylabel("Fréquence")

    plt.subplot(1,3,2)
    st.probplot(residuals, dist="norm", plot=plt.gca()); plt.title("QQ-plot des résidus")

    # Définition du seuil maximal pour les points verts sur le graphique
    THRESHOLD = 0.10

    y_test_np = np.asarray(y_test, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    # Erreur relative en gérant les zéros sur y_true
    denom = np.maximum(np.abs(y_test_np), 1e-8)
    rel_err = np.abs(y_pred_np - y_test_np) / denom

    mask_good = rel_err <= THRESHOLD
    mask_bad  = ~mask_good

    # Préparer l’axe du 3e subplot
    plt.subplot(1, 3, 3)

    # Points bons (≤ 10%) en vert
    plt.scatter(y_test_np[mask_good], y_pred_np[mask_good], s=14, alpha=0.75, color="#2ca02c", label=f"≤ {int(THRESHOLD*100)}%  (n={mask_good.sum()})")

    # Points moins bons en orange
    plt.scatter(y_test_np[mask_bad], y_pred_np[mask_bad], s=12, alpha=0.65, color="darkorange", label=f"> {int(THRESHOLD*100)}% (n={mask_bad.sum()})")

    # Droite de régression (fit sur tous les points)
    coef = np.polyfit(y_test_np, y_pred_np, deg=1)
    xline = np.linspace(y_test_np.min(), y_test_np.max(), 200)
    plt.plot(xline, coef[0]*xline + coef[1], color="black", lw=2, label=f"pente={coef[0]:.3f}")

    # Droite y = x (calibrage parfait)
    min_val = float(min(y_test_np.min(), y_pred_np.min()))
    max_val = float(max(y_test_np.max(), y_pred_np.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="y = x")

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Titre avec taux de bons points
    good_pct = 100.0 * mask_good.mean()
    plt.title(f"Réel vs Prédit (ElasticNet) — bons ≤ {int(THRESHOLD*100)}%: {good_pct:.1f}%")
    plt.xlabel("Revenu réel ($)")
    plt.ylabel("Revenu prédit ($)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()

    # Export prédictions
    # Sauvegarder dans outputs/
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({"y_test_revenue": y_test.values, "y_pred_revenue": y_pred}).to_csv(
        os.path.join("outputs", "tmdb_elasticnet_predictions_test.csv"), index=False
    )
    print("Exporté: outputs/tmdb_elasticnet_predictions_test.csv")

    # =========================================================
    # 8) p-values
    # =========================================================
    X_num_for_inference = sm.add_constant(
        df_json_parse.loc[X_train.index, ["budget_yj","popularity_log","runtime","vote_average_yj","year","month","day_of_week"]].fillna(0.0)
    )
    y_for_inference = Y.loc[X_train.index].values

    ols = sm.OLS(y_for_inference, X_num_for_inference).fit()
    print(ols.summary())

    # =========================================================
    # 9) Table des corrélations (numériques) avec la cible
    # =========================================================
    corr_feats = ["budget_yj","popularity_log","runtime","vote_average_yj","year","month","day_of_week"]
    corr_feats = [c for c in corr_feats if c in df_json_parse.columns]

    df_corr_train = df_json_parse.loc[X_train.index].copy()

    corr_yj_df = correlations_with_target(df_corr_train, feat_cols=corr_feats, target_col="revenue_yj")
    print("
=== Corrélations (Pearson/Spearman) vs revenue_yj (TRAIN) ===")
    display(corr_yj_df)
    corr_yj_df.to_csv(os.path.join("outputs", "tmdb_correlations_train_yj.csv"), index=False)
    print("Exporté: outputs/tmdb_correlations_train_yj.csv")

    corr_raw_df = correlations_with_target(df_corr_train, feat_cols=corr_feats, target_col="revenue")
    print("
=== Corrélations (Pearson/Spearman) vs revenue (TRAIN) ===")
    display(corr_raw_df)
    corr_raw_df.to_csv(os.path.join("outputs", "tmdb_correlations_train_raw.csv"), index=False)
    print("Exporté: outputs/tmdb_correlations_train_raw.csv")

    # =========================================================
    # 10) Heatmap des corrélations (optionnel, revenue_yj)
    # =========================================================
    plt.figure(figsize=(8, 5))
    num_for_corr = df_corr_train[corr_feats + ["revenue_yj"]].copy().dropna()
    corr_mat = num_for_corr.corr(method="pearson")
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matrice de corrélation (Pearson) – features numériques vs revenue_yj (TRAIN)")
    plt.tight_layout(); plt.show()
