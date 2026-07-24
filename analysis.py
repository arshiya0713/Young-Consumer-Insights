"""
Latent Consumer Insights — Young People Survey
================================================
Extracting latent psychographic drivers of young consumers' spending behaviour,
then translating them into actionable consumer personas.

Pipeline
--------
1. Load & clean the Young People Survey (1010 respondents, aged 15-30).
2. Exploratory profiling of the 7 spending-habit items.
3. Exploratory Factor Analysis (varimax) on the psychographic + interest battery
   -> recovers *latent* attitudinal dimensions respondents never state directly.
4. K-means persona segmentation on the factor scores.
5. Explanatory models: latent factors -> each spending outcome
   (which hidden attitude actually drives branded / gadget / looks / healthy spend).

All outputs (figures + tables) are written to figures/ and results/.

Author: Arshiya (GitHub: arshiya0713)
Data:   Sabo, M. (2013) "Young People Survey", FSEV UK / Kaggle.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- compat shim: newer scikit-learn renamed check_array's
#     `force_all_finite` kwarg to `ensure_all_finite`, which the pinned
#     factor_analyzer still passes. Translate it transparently. ---------------
import sklearn.utils as _sku

_orig_check_array = _sku.check_array


def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)


_sku.check_array = _check_array_compat
import factor_analyzer.factor_analyzer as _fa_mod
_fa_mod.check_array = _check_array_compat
# --------------------------------------------------------------------------- #

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_kmo,
    calculate_bartlett_sphericity,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")
RNG = 42

FIG = "figures"
RES = "results"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. LOAD & CLEAN
# --------------------------------------------------------------------------- #
df = pd.read_csv("data/responses.csv")
print(f"[load] raw shape: {df.shape}")

# The 7 spending-habit items (the marketing outcome of interest) ------------- #
SPENDING = [
    "Finances",              # "I save all the money I can" (1=disagree..5=agree)
    "Shopping centres",      # enjoyment of shopping malls
    "Branded clothing",      # willingness to pay more for a brand
    "Entertainment spending",
    "Spending on looks",
    "Spending on gadgets",
    "Spending on healthy eating",
]

# Categorical demographics / lifestyle we keep for persona profiling --------- #
CATEGORICAL = [
    "Smoking", "Alcohol", "Punctuality", "Lying", "Internet usage",
    "Gender", "Left - right handed", "Education", "Only child",
    "Village - town", "House - block of flats",
]

# Everything else numeric = the psychographic / interest battery ------------- #
numeric_cols = [c for c in df.columns if c not in CATEGORICAL]

# --- Impute: numeric -> median, categorical -> mode ------------------------- #
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())
for c in CATEGORICAL:
    df[c] = df[c].fillna(df[c].mode().iloc[0])

# The battery fed to factor analysis = numeric psychographic items,
# EXCLUDING the spending outcomes (we want latent *attitudes*, then relate
# them to spending) and raw demographics like Age/Height/Weight/#siblings.
DEMO_NUMERIC = ["Age", "Height", "Weight", "Number of siblings"]
battery = [c for c in numeric_cols if c not in SPENDING + DEMO_NUMERIC]
print(f"[load] psychographic battery items: {len(battery)}")
print(f"[load] spending outcome items:      {len(SPENDING)}")

X_bat = df[battery].copy()

# --------------------------------------------------------------------------- #
# 2. SPENDING PROFILE (descriptive)
# --------------------------------------------------------------------------- #
spend_desc = df[SPENDING].describe().T[["mean", "std", "50%"]]
spend_desc.columns = ["mean", "std", "median"]
spend_desc = spend_desc.sort_values("mean", ascending=False)
spend_desc.to_csv(f"{RES}/spending_profile.csv")
print("\n[spending] mean agreement (1=disagree .. 5=agree):")
print(spend_desc.round(2).to_string())

plt.figure(figsize=(10, 6))
order = spend_desc.index
sns.barplot(x=spend_desc["mean"], y=order, palette="crest")
plt.xlabel("Mean agreement (1–5)")
plt.ylabel("")
plt.title("Where young consumers say their money goes", pad=12)
plt.xlim(1, 5)
plt.tight_layout()
plt.savefig(f"{FIG}/01_spending_profile.png", dpi=130)
plt.close()

# Correlation among spending items ------------------------------------------- #
plt.figure(figsize=(8, 7))
sns.heatmap(df[SPENDING].corr(), annot=True, fmt=".2f", cmap="vlag",
            center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Spending-item correlations", pad=12)
plt.tight_layout()
plt.savefig(f"{FIG}/02_spending_corr.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------- #
# 3. EXPLORATORY FACTOR ANALYSIS  (the "latent needs" step)
# --------------------------------------------------------------------------- #
# Sampling adequacy checks — is the battery factorable at all?
chi_sq, p_val = calculate_bartlett_sphericity(X_bat)
kmo_all, kmo_model = calculate_kmo(X_bat)
print(f"\n[EFA] Bartlett sphericity: chi2={chi_sq:.0f}, p={p_val:.3g}")
print(f"[EFA] Kaiser–Meyer–Olkin (overall): {kmo_model:.3f}")

# Scree: eigenvalues from unrotated solution -> how many factors to keep ----- #
fa_scree = FactorAnalyzer(n_factors=X_bat.shape[1], rotation=None)
fa_scree.fit(X_bat)
ev, _ = fa_scree.get_eigenvalues()
n_kaiser = int((ev > 1).sum())
# Kaiser can over-extract on wide batteries; cap at an interpretable number.
N_FACTORS = 8
print(f"[EFA] eigenvalues>1 (Kaiser): {n_kaiser}; using N_FACTORS={N_FACTORS}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(ev) + 1), ev, "o-", ms=4)
plt.axhline(1, color="red", ls="--", lw=1, label="Kaiser (eigenvalue = 1)")
plt.axvline(N_FACTORS, color="green", ls=":", lw=1.5, label=f"retained = {N_FACTORS}")
plt.xlabel("Factor")
plt.ylabel("Eigenvalue")
plt.title("Scree plot", pad=12)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/03_scree.png", dpi=130)
plt.close()

# Fit the rotated (varimax) solution ----------------------------------------- #
fa = FactorAnalyzer(n_factors=N_FACTORS, rotation="varimax")
fa.fit(X_bat)
loadings = pd.DataFrame(
    fa.loadings_,
    index=battery,
    columns=[f"F{i+1}" for i in range(N_FACTORS)],
)

var = fa.get_factor_variance()  # (variance, proportional, cumulative)
var_tbl = pd.DataFrame(
    {"SS_loading": var[0], "prop_var": var[1], "cum_var": var[2]},
    index=loadings.columns,
)
var_tbl.to_csv(f"{RES}/factor_variance.csv")
print("\n[EFA] variance explained by retained factors:")
print(var_tbl.round(3).to_string())

# Human-readable factor labels, assigned AFTER inspecting the top-loading
# items of the fitted varimax solution (see results/factor_interpretation.txt).
FACTOR_NAMES = {
    "F1": "Highbrow Cultural Capital",   # classical/opera/jazz/art/theatre/reading
    "F2": "Fearful & Sentimental",       # phobias + romantic/celebrity/shopping
    "F3": "Sociable & Energetic",        # energy/friends/happiness/-loneliness
    "F4": "Tech, Cars & Action",         # PC/action/cars/sci-fi/science/internet
    "F5": "Conscientious & Organised",   # reliability/planning/keeping promises
    "F6": "Pop-Culture Entertainment",   # animation/fantasy/rock/movies/comedy
    "F7": "Life-Science / Medical",      # biology/chemistry/medicine/physics
    "F8": "Status & Assertiveness",      # right people/law/politics/appearance
}

# Report top-loading items per factor for interpretability ------------------- #
with open(f"{RES}/factor_interpretation.txt", "w", encoding="utf-8") as fh:
    for f in loadings.columns:
        top = loadings[f].reindex(loadings[f].abs().sort_values(ascending=False).index).head(8)
        header = f"\n=== {f}  ({FACTOR_NAMES.get(f, '')}) — {var_tbl.loc[f,'prop_var']*100:.1f}% var ==="
        print(header)
        fh.write(header + "\n")
        for item, val in top.items():
            line = f"   {val:+.2f}  {item}"
            print(line)
            fh.write(line + "\n")

# Loadings heatmap (top items only, for legibility) -------------------------- #
top_items = (
    loadings.abs().max(axis=1).sort_values(ascending=False).head(40).index
)
plt.figure(figsize=(10, 13))
sns.heatmap(
    loadings.loc[top_items],
    cmap="vlag", center=0, vmin=-1, vmax=1,
    yticklabels=True,
    xticklabels=[f"F{i+1}\n{FACTOR_NAMES[f'F{i+1}']}" for i in range(N_FACTORS)],
    cbar_kws={"shrink": .5, "label": "loading"},
)
plt.title("Varimax factor loadings (top 40 items)", pad=12)
plt.tight_layout()
plt.savefig(f"{FIG}/04_loadings.png", dpi=130)
plt.close()

# Factor scores per respondent ----------------------------------------------- #
scores = pd.DataFrame(fa.transform(X_bat), columns=loadings.columns, index=df.index)
scores_named = scores.rename(columns=FACTOR_NAMES)
scores_named.to_csv(f"{RES}/factor_scores.csv", index=False)

# --------------------------------------------------------------------------- #
# 4. PERSONA SEGMENTATION (K-means on latent factor scores)
# --------------------------------------------------------------------------- #
Xs = StandardScaler().fit_transform(scores)

sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit(Xs)
    sil[k] = silhouette_score(Xs, km.labels_)

# NOTE: silhouette is nearly flat (~0.10–0.11 across k) — psychographic space
# is continuous, so there is no single "natural" number of clusters. We
# therefore fix k for *interpretability* rather than chasing a marginal max.
# The robust finding is the factor structure (KMO=0.81); personas are an
# actionable, deliberately-soft partition of that continuous space.
K_PERSONAS = 5
best_k = K_PERSONAS
print("\n[persona] silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
print(f"[persona] silhouette is nearly flat -> fixing k = {best_k} for interpretability")

plt.figure(figsize=(9, 5))
plt.plot(list(sil.keys()), list(sil.values()), "o-")
plt.axvline(best_k, color="green", ls=":", label=f"k = {best_k} (chosen)")
plt.xlabel("Number of personas (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette is flat — no single natural k", pad=12)
plt.ylim(0, max(sil.values()) * 1.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/05_silhouette.png", dpi=130)
plt.close()

km = KMeans(n_clusters=best_k, random_state=RNG, n_init=10).fit(Xs)
df["persona"] = km.labels_
scores_named["persona"] = km.labels_

# Persona x latent-factor mean profile --------------------------------------- #
persona_profile = scores_named.groupby("persona").mean()
persona_profile.to_csv(f"{RES}/persona_factor_profile.csv")

plt.figure(figsize=(11, 6))
sns.heatmap(persona_profile.T, cmap="vlag", center=0, annot=True, fmt=".2f",
            cbar_kws={"label": "mean factor score (z)"})
plt.xlabel("Persona")
plt.ylabel("")
plt.title("Persona fingerprints across latent factors", pad=12)
plt.tight_layout()
plt.savefig(f"{FIG}/06_persona_factor_heatmap.png", dpi=130)
plt.close()

# Persona x spending behaviour ----------------------------------------------- #
persona_spend = df.groupby("persona")[SPENDING].mean()
persona_spend.to_csv(f"{RES}/persona_spending.csv")

plt.figure(figsize=(11, 6))
sns.heatmap(persona_spend.T, cmap="crest", annot=True, fmt=".2f",
            cbar_kws={"label": "mean agreement (1–5)"})
plt.xlabel("Persona")
plt.ylabel("")
plt.title("How each persona spends", pad=12)
plt.tight_layout()
plt.savefig(f"{FIG}/07_persona_spending.png", dpi=130)
plt.close()

# Persona sizes + a couple of demographic descriptors ------------------------ #
persona_summary = df.groupby("persona").agg(
    n=("Age", "size"),
    mean_age=("Age", "mean"),
    pct_female=("Gender", lambda s: (s == "female").mean() * 100),
)
persona_summary["pct_of_sample"] = persona_summary["n"] / len(df) * 100
persona_summary.to_csv(f"{RES}/persona_summary.csv")
print("\n[persona] summary:")
print(persona_summary.round(1).to_string())

# --------------------------------------------------------------------------- #
# 5. WHAT LATENT ATTITUDE DRIVES SPENDING?  (explanatory regressions)
# --------------------------------------------------------------------------- #
# For each spending outcome, regress it on the 8 latent factor scores.
# Standardised factor scores -> coefficients are directly comparable.
Xf = StandardScaler().fit_transform(scores)   # scores columns are F1..F8
factor_labels = [FACTOR_NAMES[c] for c in scores.columns]
driver_rows = []
for outcome in SPENDING:
    y = df[outcome].values
    lr = LinearRegression().fit(Xf, y)
    r2 = lr.score(Xf, y)
    for name, coef in zip(factor_labels, lr.coef_):
        driver_rows.append({"outcome": outcome, "factor": name,
                            "beta": coef, "model_R2": r2})

drivers = pd.DataFrame(driver_rows)
drivers.to_csv(f"{RES}/spending_drivers.csv", index=False)

driver_wide = drivers.pivot(index="factor", columns="outcome", values="beta")
plt.figure(figsize=(11, 7))
sns.heatmap(driver_wide, cmap="vlag", center=0, annot=True, fmt=".2f",
            cbar_kws={"label": "std. regression coefficient"})
plt.title("Latent attitudes → spending (standardised β)", pad=12)
plt.xlabel("")
plt.ylabel("Latent factor")
plt.tight_layout()
plt.savefig(f"{FIG}/08_spending_drivers.png", dpi=130)
plt.close()

r2_tbl = drivers.groupby("outcome")["model_R2"].first().sort_values(ascending=False)
print("\n[drivers] variance in spending explained by latent attitudes (R^2):")
print(r2_tbl.round(3).to_string())

print("\n[done] figures -> figures/ , tables -> results/")
