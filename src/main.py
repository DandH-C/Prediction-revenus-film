# Imports
import warnings
warnings.filterwarnings("ignore")

import os, json, re, unicodedata, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import skew, pearsonr, spearmanr
from matplotlib.ticker import ScalarFormatter

import kagglehub
import statsmodels.api as sm
import argparse

from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Helper Plotly pour afficher les tableaux (DataFrame/Series) ---
def show_plotly_table(obj, title=None, max_rows=1000, max_cols=40):
    if isinstance(obj, pd.Series):
        df_tbl = obj.to_frame(name=(obj.name or "value")).reset_index()
        # Renommer proprement la première colonne si nom étrange
        if not isinstance(df_tbl.columns[0], str) or df_tbl.columns[0] == "index":
            df_tbl.columns = ["index", df_tbl.columns[1]]
    elif isinstance(obj, pd.DataFrame):
        df_tbl = obj.reset_index()
    else:
        return  # on ignore les objets non tabulaires

    df_tbl = df_tbl.iloc[:max_rows, :max_cols]
    header_values = [str(c) for c in df_tbl.columns]
    cells_values = [df_tbl[c].astype(str).tolist() for c in df_tbl.columns]

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, fill_color="#f2f2f2", align="left"),
        cells=dict(values=cells_values, align="left"))
    ])
    fig.update_layout(title=title or "", height=min(120 + 24*len(df_tbl), 1000),
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.show()

# Paramètres
RANDOM_STATE = 42
TARGET_COL = "revenue"

JSON_COLS = ["genres", "keywords", "production_companies", "production_countries", "spoken_languages", "cast", "crew", "title"]

# Valeurs min/max pour la sélection des mots
MIN_DF, MAX_DF = 6, 0.85
TOP_N_CAST, TOP_N_CREW = 180, 180
TITLE_MAX_FEATURES = 15000

# Compression SVD pour éviter les blocs dense
SVD_PER_BLOCK = {"cast":180, "crew":180, "kw":450, "genres":20, "pc":300, "cty":40, "lang":40, "title":200}

# Paramètres K-fold
N_FOLDS = 8    # Pour une exécution rapide mais moins précise, choisir entre 3 et 8
N_ITER  = 150  # Ici, entre 40 et 250


parser = argparse.ArgumentParser(description="TMDb ElasticNet pipeline (local/Kaggle)")
parser.add_argument("--data-source", choices=["kaggle", "local"], default="kaggle", help="Source des données: 'kaggle' (via kagglehub) ou 'local' (répertoire avec CSV)")
parser.add_argument("--data-dir", type=str, default="data", help="Répertoire local contenant tmdb_5000_movies.csv et tmdb_5000_credits.csv")
args, _ = parser.parse_known_args()

# =========================================================
# 1) Chargement des données (kagglehub ou local)
# =========================================================
def load_from_local(data_dir: str):
    """Lit les CSV depuis un répertoire local."""
    movies_path  = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(
            f"Fichiers introuvables dans '{data_dir}'. "
            f"Placez 'tmdb_5000_movies.csv' et 'tmdb_5000_credits.csv'."
        )
    print(f"[LOCAL] Lecture: {movies_path} | {credits_path}")
    return pd.read_csv(movies_path), pd.read_csv(credits_path)

def load_from_kagglehub():
    """Télécharge via kagglehub et lit les CSV."""
    print("[KAGGLEHUB] Téléchargement du dataset…")
    folder = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print("Téléchargement KaggleHub ->", folder)
    return (pd.read_csv(os.path.join(folder, "tmdb_5000_movies.csv")),
            pd.read_csv(os.path.join(folder, "tmdb_5000_credits.csv")))

# Choix de la source (avec fallback vers local si Kaggle échoue)
if args.data_source == "local":
    df_movies, df_credits = load_from_local(args.data_dir)
else:
    try:
        df_movies, df_credits = load_from_kagglehub()
    except Exception as e:
        print(f"[WARN] KaggleHub a échoué ({type(e).__name__}: {e}). "
              f"Basculer sur données locales si disponibles…")
        df_movies, df_credits = load_from_local(args.data_dir)

# =========================================================
# 2) Préparation / Filtrage
# =========================================================
# Forcer type numérique
for c in ["budget", "revenue", "runtime"]:
    if c in df_movies.columns:
        df_movies[c] = pd.to_numeric(df_movies[c], errors="coerce")

# Vérifier les colonnes title si elles sont identiques
col1 = df_movies["title"]; col2 = df_credits["title"]
# Décommenter pour afficher
# print("\nLes 2 colonnes title sont identiques?", col1.equals(col2))

# Drop 'title' côté crédits
df_credits = df_credits.drop(columns=["title"], errors="ignore")

# Fusionner les 2 datasets
df = df_movies.merge(df_credits, left_on="id", right_on="movie_id", how="inner")

# Colonnes à supprimer pour simplifier le modèle
drop_cols = ["id", "movie_id", "homepage", "original_language", "status", "tagline", "overview", "original_title", "vote_count"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Masque pour budget <=0 ou revenue <=0
bud_rev_0 = (df["budget"] <= 0) | (df["revenue"] <= 0)

# Décommenter pour afficher, utiliser lors de l'analyse
# print("Avant filtre: budget <= 0 OU revenue <= 0:", int(bud_rev_0.sum()), "lignes sur", len(df))
df = df.loc[~bud_rev_0].copy()

# Décommenter pour afficher
# print("\nAprès filtre, les variables revenue et budget:")
# print("% revenue == 0:", (df["revenue"] == 0).mean() * 100, "| % budget == 0:", (df["budget"] == 0).mean() * 100)

# Création des colonnes en numérique pour la date
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["year"] = df["release_date"].dt.year
df["month"] = df["release_date"].dt.month
df["day_of_week"] = df["release_date"].dt.dayofweek
df = df.drop(columns=["release_date"])


# =========================================================
# 3) Variable cible + EDA initiale (tout en une seule figure)
# =========================================================
y_raw = pd.to_numeric(df["revenue"], errors="coerce")
mask = y_raw.notna()
y = y_raw[mask].to_numpy().reshape(-1, 1)

# Transformation Yeo-Johnson
y_yj = PowerTransformer(method="yeo-johnson").fit_transform(y)

# Ajout colonne revenue_yj
df["revenue_yj"] = np.nan
df.loc[mask, "revenue_yj"] = y_yj.flatten()

# Préparation pour l'affichage
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
(ax1, ax2), (ax3, ax4) = axes

# Histogramme de l'asymétrie de la variable brut
ax1.hist(y_raw.dropna(), bins=50, color="steelblue", alpha=0.8)
ax1.set_title(f"Distribution des revenus avant transformation. Asymétrie: {y_raw.skew():.3f}")
ax1.set_xlabel("Revenue"); ax1.set_ylabel("Fréquences")
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax1.ticklabel_format(style="plain", axis="y")

# Histogramme de la variable transformée YJ
ax2.hist(y_yj, bins=50, color="darkorange", alpha=0.8)
ax2.set_title(f"Distribution des revenus après transormation Yeo-Johnson. Asymétrie: {skew(y_yj.flatten(), bias=False):.3f}")
ax2.set_xlabel("revenus_YJ"); ax2.set_ylabel("Fréquences")
ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax2.ticklabel_format(style="plain", axis="y")

# QQ-plot: revenue brut
st.probplot(df["revenue"].dropna(), dist="norm", plot=ax3)
ax3.set_title("QQ-plot: revenue brut")

# QQ-plot: revenue YJ
st.probplot(df["revenue_yj"].dropna(), dist="norm", plot=ax4)
ax4.set_title("QQ-plot: revenue YJ")

plt.tight_layout()
plt.show()


# Shape du dataframe
# print(f"Shape finale: {df.shape}")

# Voir si nous avons des valeurs absentes
#na_tbl = df.isna().sum().rename("NaN")
#show_plotly_table(na_tbl, title="Valeurs nulles ou NaN par colonne")


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
# print("Problèmes potentiels:")
# print(table_probleme)

# print("\nAffichage des données parsé json")
# print("\n", df_json_parse.head())

# =========================================================
# 5) Analyse sur les variables numériques
# =========================================================
num_col = ["budget", "popularity", "runtime", "vote_average", "year", "month", "day_of_week"]

#print(df_json_parse[num_col].describe())
_desc_num = df_json_parse[num_col].describe()
show_plotly_table(_desc_num, title="Statistiques descriptives – variables numériques")

# Catégoriser revenue_yj en quantiles pour boxplots
df_json_parse["revenue_category"] = pd.qcut(df_json_parse["revenue_yj"], q=4, duplicates="drop")
df_json_parse["revenue_category"] = df_json_parse["revenue_category"].apply(lambda x: pd.Interval(round(x.left, 2), round(x.right, 2)))


from math import ceil

cats_order = sorted(df_json_parse["revenue_category"].dropna().unique(), key=lambda iv: (iv.left, iv.right))
n = len(num_col)
rows = ceil(n / 3)
fig, axes = plt.subplots(rows, 3, figsize=(18, 4.8*rows), constrained_layout=True)
axes = np.array(axes).reshape(-1)  # à plat pour indexer facilement

for i, col in enumerate(num_col):
    ax = axes[i]
    sns.boxplot(
        x="revenue_category", y=col,
        data=df_json_parse,
        order=cats_order,
        palette="Set2",
        showfliers=False,   # évite les points extrêmes qui compressent les boîtes
        width=0.6,          # boîtes un peu plus fines
        ax=ax
    )
    ax.set_title(f"{col} vs revenue Yeo Johnson")
    ax.set_xlabel("Revenue Categorie")
    ax.set_ylabel(col)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)  # meilleure lisibilité

# masque les axes vides si n < rows*3
for j in range(i+1, rows*3):
    fig.delaxes(axes[j])

plt.show()


# Création des asymétries des colonnes numériques originales
sk_num = df_json_parse[num_col].skew(numeric_only=True)
show_plotly_table(sk_num.sort_values(ascending=False).rename("skew"), title="Asymétrie des variables numériques")

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

cats_order = sorted(df_json_parse["revenue_category"].dropna().unique(), key=lambda iv: (iv.left, iv.right))
n2 = len(new_num_col)
rows2 = ceil(n2 / 3)
fig2, axes2 = plt.subplots(rows2, 3, figsize=(18, 4.8*rows2), constrained_layout=True)
axes2 = np.array(axes2).reshape(-1)

for i, col in enumerate(new_num_col):
    ax = axes2[i]
    sns.boxplot(
        x="revenue_category", y=col,
        data=df_json_parse,
        order=cats_order,
        palette="Set2",
        showfliers=False,
        width=0.6,
        ax=ax
    )
    ax.set_title(f"{col} vs revenue Yeo Johnson")
    ax.set_xlabel("Revenu Categorie")
    ax.set_ylabel(col)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

for j in range(i+1, rows2*3):
    fig2.delaxes(axes2[j])

plt.show()


# Création des asymétrie des colonnes numériques transformées
sk_new = df_json_parse[new_num_col].skew(numeric_only=True).sort_values(ascending=False)
show_plotly_table(sk_new.rename("skew"), title="Asymétrie des variables numériques transformées")

# =========================================================
# 7) Modélisation
# =========================================================
# Définition des jeux d'entraînement et de test
Y = df_json_parse[TARGET_COL].astype(float)
X = df_json_parse.drop(columns=[TARGET_COL])

# 20% test et 80% entrainement
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=RANDOM_STATE)
# print("Split:", X_train.shape, X_test.shape)

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
    # Si l'on desire afficher sur quelle colonne nous sommes rendu avec le TF-IDF
 #   print(f"[TF-IDF] {col}")
    as_title = (col == "title")
    Xtr_block, Xte_block = tfidf_fit_transform(X_train[col], X_test[col], as_title=as_title)

    # Sélection des Top-N pour cast/crew pour limiter la variabilité
    top_n = TOP_N_CAST if col == "cast" else (TOP_N_CREW if col == "crew" else None)
    if top_n:
        sums = np.asarray(Xtr_block.sum(axis=0)).ravel()
        order = np.argsort(-sums)
        keep_idx = order[:min(top_n, Xtr_block.shape[1])]
        Xtr_block = Xtr_block[:, keep_idx]; Xte_block = Xte_block[:, keep_idx]

    # Compression SVD par bloc json
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

# print(f"\nTF-IDF Train: {Xtr_tfidf.shape} | Test: {Xte_tfidf.shape}")

# Attributs numérique, transformation YJ pour train et test
def transform_numeric_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame):
    tr, te = df_train.copy(), df_test.copy()

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    num_tr = pd.DataFrame({"budget": pd.to_numeric(tr.get("budget_yj", tr.get("budget", 0.0)), errors="coerce"),
        "vote_average": pd.to_numeric(tr.get("vote_average_yj", tr.get("vote_average", 0.0)), errors="coerce")}).fillna(0.0)
    num_te = pd.DataFrame({"budget": pd.to_numeric(te.get("budget_yj", te.get("budget", 0.0)), errors="coerce"),
        "vote_average": pd.to_numeric(te.get("vote_average_yj", te.get("vote_average", 0.0)), errors="coerce")}).fillna(0.0)

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
            rows.append({"feature": col, "pearson_r": np.nan, "pearson_pValue": np.nan,
                         "spearman_rho": np.nan, "spearman_pValue": np.nan, "n": int(valid.sum())})
            continue

        # Calcul des corrélations + p-values linéaires et monotone
        r,  p  = pearsonr(x[valid], y[valid])
        rs, ps = spearmanr(x[valid], y[valid])
        rows.append({"feature": col, "pearson_r": r, "pearson_pValue": p,
                     "spearman_rho": rs, "spearman_pValue": ps, "n": int(valid.sum())})

    # Assemblage du df
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_pearson"] = out["pearson_r"].abs()
        out = out.sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"])
    # Tableau contenant les colonnes features, pearson_r, pearson_pValue, spearman_rho, spearman_pValue et n.
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
prep = ColumnTransformer(transformers=[("tfidf_comp", StandardScaler(with_mean = False), tfidf_cols), ("num_core", core_pipe, core_num)], remainder="drop")

# Création du modèle
base = Pipeline([("preprocessor", prep), ("clf", ElasticNet(random_state=RANDOM_STATE, max_iter=15000))])

target_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)

# Réduction de l'asymétrie avec YJ lors de l'entraînement et retour en échelle normale pour les revenus
ttr = TransformedTargetRegressor(regressor=base, transformer=PowerTransformer(method="yeo-johnson", standardize=True))

# Hyperparamètres
param_dist = {
    "regressor__clf__alpha": np.unique(np.concatenate([
              np.logspace(-4.5, -0.5, 60),
              np.linspace(0.005, 0.10, 60)])),

    "regressor__clf__l1_ratio": np.linspace(0.1, 0.9, 10)}

# CV par quantiles sur y_train avec K-Fold
q = pd.qcut(y_train.rank(method="first"), q=N_FOLDS, labels=False, duplicates="drop")
cv_splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y_train)), q))

# Constructeur search pour RandomizedSearchCV
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
print("Meilleurs paramètres:", search.best_params_)
print(f"Meilleur RMSE CV: {cv_rmse:.2f}")

# Définition de l'erreur symétrique entre y_true et y_pred (réduction des erreurs avec des valeurs près de 0)
# Obtenir la plus petite valeur possible
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)); denom[denom == 0] = 1.0
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

# Début des prédictions et RMSE dans la même unité que la cible
def evaluate(estimator, Xte, yte, ytr, title="ElasticNet"):
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

    
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "SMAPE (%)": s_mape,
        "RMSE de base": rmse_base,
        "Gain vs base (%)": gain,
    }
    df_tbl = pd.DataFrame(metrics, index=[title]).T.reset_index()
    df_tbl.columns = ["Mesure", "Valeur"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_tbl.columns),
                    fill_color="#f2f2f2", align="left"),
        cells=dict(values=[df_tbl[c].astype(str).tolist() for c in df_tbl.columns],
                   align="left")
    )])
    fig.update_layout(title=f"Évaluation {title} – Tableau des métriques",
                      height=min(120 + 28*len(df_tbl), 700),
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.show()


    # Print des résultats
   # print(f"\nÉvaluation {title}")
   # print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | SMAPE: {s_mape:.2f}%")
   # print(f"RMSE de base: {rmse_base:.2f} | Gain vs base: {gain:.1f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "smape": s_mape, "Base rmse": rmse_base, "gain en %": gain}

# Exécution des métriques pour trouver le meilleur ajustement possible
test_metrics = evaluate(search.best_estimator_, Xte_all, y_test, y_train, title="ElasticNet")

# Plots histogramme des résidus
y_pred = search.best_estimator_.predict(Xte_all)
residuals = y_test - y_pred
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.hist(residuals, bins=40, color="steelblue", alpha=0.85)
plt.title("Histogramme des résidus"); plt.xlabel("Résidu ($)"); plt.ylabel("Fréquence")

# QQ plots des résidus
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
plt.scatter(y_test_np[mask_good], y_pred_np[mask_good], s=14, alpha=0.75, color="green", label=f"≤ {int(THRESHOLD*100)}%  (n={mask_good.sum()})")

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

# Export des prédictions
pd.DataFrame({"y_test_revenue": y_test.values, "y_pred_revenue": y_pred}).to_csv(
    "tmdb_elasticnet_predictions_test.csv", index=False
)
print("Exporté: tmdb_elasticnet_predictions_test.csv")

# =========================================================
# 8) Création des p-values pour validation des hypothèses
# =========================================================
X_num_for_inference = sm.add_constant(
    df_json_parse.loc[X_train.index, ["budget_yj","popularity_log","runtime","vote_average_yj","year","month","day_of_week"]].fillna(0.0))
y_for_inference = Y.loc[X_train.index].values

ols = sm.OLS(y_for_inference, X_num_for_inference).fit()
print(ols.summary())
# Affichage Plotly des coefficients OLS
_ols_tbl = pd.concat([
    ols.params.rename("coef"),
    ols.bse.rename("std err"),
    ols.tvalues.rename("t"),
    ols.pvalues.rename("P>|t|"),
], axis=1)
_ci = ols.conf_int()
_ols_tbl["[0.025"] = _ci[0]
_ols_tbl["0.975]"] = _ci[1]
show_plotly_table(_ols_tbl, title="Coefficients")

# =========================================================
# 9) Table des corrélations numériques avec la cible
# =========================================================
corr_feats = ["budget_yj","popularity_log","runtime","vote_average_yj","year","month","day_of_week"]
corr_feats = [c for c in corr_feats if c in df_json_parse.columns]

df_corr_train = df_json_parse.loc[X_train.index].copy()

corr_yj_df = correlations_with_target(df_corr_train, feat_cols=corr_feats, target_col="revenue_yj")
show_plotly_table(corr_yj_df, title="Corrélations vs revenue_yj (TRAIN)")

# Export corrélation avec valeurs transformées
corr_yj_df.to_csv("tmdb_correlations_train_yj.csv", index=False)
print("Exporté: tmdb_correlations_train_yj.csv")

corr_raw_df = correlations_with_target(df_corr_train, feat_cols=corr_feats, target_col="revenue")
show_plotly_table(corr_raw_df, title="Corrélations vs revenue (TRAIN)")

# Export du csv des corrélations avec valeurs raw
corr_raw_df.to_csv("tmdb_correlations_train_raw.csv", index=False)
print("Exporté: tmdb_correlations_train_raw.csv")

# =========================================================
# 10) Heatmap des corrélations - Décommentez pour afficher la matrice
# =========================================================
#plt.figure(figsize=(8, 5))
#num_for_corr = df_corr_train[corr_feats + ["revenue_yj"]].copy().dropna()
#corr_mat = num_for_corr.corr(method="pearson")
#sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
#plt.title("Matrice de corrélation (Pearson) – features numériques vs revenue_yj (TRAIN)")
#plt.tight_layout(); plt.show()
