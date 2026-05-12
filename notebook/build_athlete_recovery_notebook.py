"""Build athlete_recovery_prediction.ipynb — uçtan uca regresyon projesi.

Veri: https://www.kaggle.com/datasets/sarveshchhetri/athlete-recovery-and-biometric-performance-dataset

Yerel CSV (athlete_recovery_synthetic.csv) varsa kullanılır.
Yoksa kagglehub ile indirilir; o da olmazsa küçük sentetik örnek üretilir.
"""
import json
from textwrap import dedent

NB_PATH = "athlete_recovery_prediction.ipynb"


def md(src: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(src).strip("\n").splitlines(keepends=True),
    }


def code(src: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(src).strip("\n").splitlines(keepends=True),
    }


cells = []

cells.append(md("""
# Atlet Toparlanma Skoru Tahmini — Uçtan Uca Veri Bilimi Projesi (Regresyon)

**Veri kaynağı:** [Athlete Recovery & Biometric Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/sarveshchhetri/athlete-recovery-and-biometric-performance-dataset)

Sentetik fakat gerçekçi bir longitudinal veri: **~300 atlet × 28 gün** boyunca antrenman, biyometri ve yaşam tarzı kayıtları. Hedef `Recovery_Score` (0–100) **sürekli** bir değişken olduğundan görev **regresyon**.

Bu notebook şu adımları tek akışta uygular:
1. Veri yükleme (yerel CSV → kagglehub → örnek üret),
2. Veri kalitesi ve eksik değer analizi,
3. Hedef ve değişken EDA'sı,
4. Train / val / test ayrımı (athlete-bazlı opsiyon),
5. Özellik mühendisliği (Training_Load, Sleep_Deficit, lag/rolling),
6. Preprocessing pipeline (impute + scale + one-hot),
7. Çoklu regresyon modeli karşılaştırması,
8. Çapraz doğrulama, RandomizedSearchCV ile hiperparametre ayarı,
9. Test değerlendirmesi, artık (residual) analizi, feature importance,
10. Model kaydetme ve örnek tahmin.

**Veri dosyası:** Aşağıdaki adlardan biriyle proje klasörüne CSV koyun: `athlete_recovery_synthetic.csv` (veya `athlete_recovery.csv`, `data.csv`). Yoksa `kagglehub` ile indirme denenir; o da olmazsa örnek veri üretilir.
"""))

cells.append(md("## 1) Kurulum ve kütüphaneler"))
cells.append(code("""
import warnings
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split, cross_val_score, RandomizedSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 60)
print("Hazir. random_state =", RANDOM_STATE)
"""))

cells.append(md("## 2) Veri yükleme (yerel CSV → kagglehub → örnek)"))
cells.append(code("""
LOCAL_NAMES = [
    "athlete_recovery_synthetic.csv",
    "athlete_recovery.csv",
    "Athlete_Recovery.csv",
    "data.csv",
]

def resolve_csv():
    root = Path(".").resolve()
    for name in LOCAL_NAMES:
        p = root / name
        if p.exists():
            return p
    try:
        import kagglehub
        dl = Path(kagglehub.dataset_download("sarveshchhetri/athlete-recovery-and-biometric-performance-dataset"))
        csvs = list(dl.rglob("*.csv"))
        if csvs:
            return max(csvs, key=lambda x: x.stat().st_size)
    except Exception as e:
        print("kagglehub indirme atlandi:", type(e).__name__, str(e)[:120])
    return None


def make_sample_data(n_athletes: int = 60, days: int = 28, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sports = ["Team Sport", "Endurance", "Strength", "Combat", "Mixed"]
    trainings = ["HIIT", "Cardio", "Yoga", "Strength", "Rest"]
    genders = ["Male", "Female", "Non-binary"]
    stress_levels = ["Low", "Medium", "High"]
    for ath in range(n_athletes):
        age = int(rng.integers(18, 38))
        gender = rng.choice(genders, p=[0.55, 0.42, 0.03])
        sport = rng.choice(sports)
        for d in range(1, days + 1):
            tr_type = rng.choice(trainings, p=[0.2, 0.25, 0.15, 0.2, 0.2])
            dur = 0 if tr_type == "Rest" else int(rng.integers(20, 120))
            intensity = 0.0 if tr_type == "Rest" else float(np.round(rng.uniform(2, 10), 1))
            sleep = float(np.round(rng.normal(7, 1.2), 2))
            caffeine = int(max(0, rng.normal(150, 80)))
            stress = rng.choice(stress_levels, p=[0.35, 0.45, 0.2])
            rhr = int(np.clip(rng.normal(62, 8) + (intensity * 0.5), 40, 110))
            hrv = int(np.clip(rng.normal(70, 18) - (intensity * 1.2), 20, 140))
            mood = float(np.round(rng.uniform(1, 10), 1))
            sore = float(np.round(np.clip(rng.normal(intensity * 0.6, 1.5), 0, 10), 1))
            energy = float(np.round(np.clip(10 - sore + rng.normal(0, 1.2), 1, 10), 1))
            # Recovery_Score sentetik formul (gercege benzer karisik etki)
            recovery = (
                50
                + 4.0 * (sleep - 7)
                + 0.15 * (hrv - 70)
                - 0.4 * (rhr - 62)
                - 2.0 * intensity * (dur / 60)
                - 1.5 * (1 if stress == "High" else (0.6 if stress == "Medium" else 0))
                + 1.2 * (mood - 5)
                - 1.8 * sore
                + 1.0 * (energy - 5)
                + rng.normal(0, 4)
            )
            recovery = float(np.clip(np.round(recovery, 1), 0, 100))
            rows.append({
                "Athlete_ID": 1000 + ath,
                "Day": d,
                "Day_of_Week": ((d - 1) % 7) + 1,
                "Week": ((d - 1) // 7) + 1,
                "Age": age,
                "Gender": gender,
                "Sport_Type": sport,
                "Training_Type": tr_type,
                "Training_Duration_Min": dur,
                "Training_Intensity": intensity,
                "Sleep_Duration_Hours": sleep,
                "Caffeine_Intake_mg": caffeine,
                "Stress_Level": stress,
                "Resting_Heart_Rate": rhr,
                "HRV_ms": hrv,
                "Mood_Score": mood,
                "Muscle_Soreness": sore,
                "Energy_Level": energy,
                "Recovery_Score": recovery,
            })
    out = pd.DataFrame(rows)
    # Az miktarda eksik veri ekle (gercek veri ile uyumlu olmasi icin)
    miss_idx = rng.choice(out.index, size=int(len(out) * 0.07), replace=False)
    out.loc[miss_idx, "Sleep_Duration_Hours"] = np.nan
    return out


path = resolve_csv()
if path is not None:
    df = pd.read_csv(path)
    print("Veri dosyasi:", path)
else:
    print("Yerel CSV / kagglehub yok; ornek veri uretiliyor (gercek analiz icin Kaggle CSV yukleyin).")
    df = make_sample_data()

print("Boyut:", df.shape)
df.head()
"""))

cells.append(md("## 3) Hedef ve sütun tanımı"))
cells.append(code("""
TARGET = "Recovery_Score"
GROUP_COL = "Athlete_ID" if "Athlete_ID" in df.columns else None
ID_LIKE = {"athlete_id", "id", "index", "unnamed: 0"}

assert TARGET in df.columns, f"Hedef sutun '{TARGET}' bulunamadi."

print("Hedef:", TARGET)
print("Hedef ozet:")
print(df[TARGET].describe())
print("\\nAtlet sayisi:", df[GROUP_COL].nunique() if GROUP_COL else "N/A")
print("Toplam gun gozlemi:", len(df))
"""))

cells.append(md("## 4) Veri kalitesi"))
cells.append(code("""
df.info()
print("\\nEksik deger (sirali):")
print(df.isnull().sum().sort_values(ascending=False).head(15))
print("\\nTekrar satir:", df.duplicated().sum())
print("\\nKategorik sutun benzersizleri:")
for c in df.select_dtypes(include="object").columns:
    print(f"  {c}: {sorted(df[c].dropna().unique().tolist())[:10]}")
"""))

cells.append(md("## 5) Hedef dağılımı"))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df[TARGET].dropna(), kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title(f"{TARGET} dagilimi")
sns.boxplot(x=df[TARGET].dropna(), ax=axes[1], color="#55A868")
axes[1].set_title(f"{TARGET} boxplot")
plt.tight_layout()
plt.show()

print("Skewness:", round(df[TARGET].skew(), 3))
print("Kurtosis:", round(df[TARGET].kurt(), 3))
"""))

cells.append(md("## 6) Sayısal değişkenlerin hedefle ilişkisi (korelasyon)"))
cells.append(code("""
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in {"Day", "Day_of_Week", "Week", "Athlete_ID"}]

plt.figure(figsize=(11, 8))
sns.heatmap(df[num_cols].corr(numeric_only=True), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 8})
plt.title("Sayisal degisken korelasyon matrisi")
plt.tight_layout()
plt.show()

corr_with_target = df[num_cols].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
print("Hedefle korelasyonlar (|r| sirali):")
print(corr_with_target.round(3))
"""))

cells.append(md("## 7) Kategorik değişkenlerin hedefe etkisi"))
cells.append(code("""
cat_cols = df.select_dtypes(include="object").columns.tolist()
if cat_cols:
    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, cat_cols):
        order = df.groupby(c)[TARGET].median().sort_values().index
        sns.boxplot(data=df, x=c, y=TARGET, ax=ax, order=order)
        ax.set_title(f"{TARGET} ~ {c}")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()

    for c in cat_cols:
        print(f"\\n{c} bazinda ortalama {TARGET}:")
        print(df.groupby(c)[TARGET].agg(["mean", "median", "count"]).round(2).sort_values("mean", ascending=False))
"""))

cells.append(md("## 8) Atletler arası ve haftalar arası örüntü"))
cells.append(code("""
if GROUP_COL is not None:
    n_sample = min(20, df[GROUP_COL].nunique())
    sampled_ids = pd.Series(df[GROUP_COL].unique()).sample(n_sample, random_state=0).tolist()
    sub = df[df[GROUP_COL].isin(sampled_ids)].copy()
    sub["Athlete_Sample"] = sub[GROUP_COL].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(data=sub, x="Athlete_Sample", y=TARGET, ax=axes[0])
    axes[0].set_title(f"Ornek {n_sample} atlet — Recovery_Score dagilimi")
    axes[0].tick_params(axis="x", rotation=80)

    if "Week" in df.columns:
        weekly = df.groupby("Week")[TARGET].mean().reset_index()
        sns.lineplot(data=weekly, x="Week", y=TARGET, marker="o", ax=axes[1])
        axes[1].set_title("Haftalik ortalama Recovery_Score")
    else:
        axes[1].axis("off")
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("## 9) Train / val / test ayrımı (atlet bazlı opsiyonel)"))
cells.append(code("""
# Athlete-bazli split: ayni atletin farkli gunleri ayni split icinde kalir → veri sizintisini azaltir.
SPLIT_MODE = "athlete"  # "random" yapilirsa rastgele satir bazli olur.

feat_cols = [c for c in df.columns if c not in {TARGET, GROUP_COL}]
X_all = df[feat_cols].copy()
y_all = df[TARGET].astype(float).copy()

if SPLIT_MODE == "athlete" and GROUP_COL is not None:
    groups = df[GROUP_COL].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    tv_idx, te_idx = next(gss1.split(X_all, y_all, groups))
    X_tv, X_test = X_all.iloc[tv_idx], X_all.iloc[te_idx]
    y_tv, y_test = y_all.iloc[tv_idx], y_all.iloc[te_idx]
    groups_tv = groups[tv_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(gss2.split(X_tv, y_tv, groups_tv))
    X_train, X_val = X_tv.iloc[tr_idx], X_tv.iloc[va_idx]
    y_train, y_val = y_tv.iloc[tr_idx], y_tv.iloc[va_idx]
else:
    X_tv, X_test, y_tv, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1765, random_state=RANDOM_STATE)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
if GROUP_COL is not None and SPLIT_MODE == "athlete":
    tr_g = set(df.loc[X_train.index, GROUP_COL].unique())
    va_g = set(df.loc[X_val.index, GROUP_COL].unique())
    te_g = set(df.loc[X_test.index, GROUP_COL].unique())
    print("Train atlet:", len(tr_g), "| Val atlet:", len(va_g), "| Test atlet:", len(te_g))
    print("Kesisim (train∩test):", len(tr_g & te_g))
"""))

cells.append(md("## 10) Özellik mühendisliği"))
cells.append(code("""
def add_features(din: pd.DataFrame) -> pd.DataFrame:
    d = din.copy()
    # Antrenman yuku = Sure (saat) x Yogunluk
    if "Training_Duration_Min" in d.columns and "Training_Intensity" in d.columns:
        d["Training_Load"] = (d["Training_Duration_Min"].fillna(0) / 60.0) * d["Training_Intensity"].fillna(0)
    # Uyku acigi (8 saate gore)
    if "Sleep_Duration_Hours" in d.columns:
        d["Sleep_Deficit"] = (8.0 - d["Sleep_Duration_Hours"]).clip(lower=0)
    # HRV / RHR orani (otonom denge proxy'si)
    if "HRV_ms" in d.columns and "Resting_Heart_Rate" in d.columns:
        d["HRV_RHR_Ratio"] = d["HRV_ms"] / d["Resting_Heart_Rate"].replace(0, np.nan)
    # Hafta ici / hafta sonu
    if "Day_of_Week" in d.columns:
        d["Is_Weekend"] = (d["Day_of_Week"] >= 6).astype(int)
    return d


X_train_fe = add_features(X_train)
X_val_fe = add_features(X_val)
X_test_fe = add_features(X_test)

print("Yeni sutunlar:")
print([c for c in X_train_fe.columns if c not in X_train.columns])
X_train_fe.head()
"""))

cells.append(md("## 11) Preprocessing pipeline"))
cells.append(code("""
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

num_features = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
cat_features = [c for c in X_train_fe.columns if c not in num_features]
print("Sayisal:", num_features)
print("Kategorik:", cat_features)

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", make_onehot())]), cat_features),
    ]
)


def build_reg(model):
    return Pipeline([("pre", pre), ("model", model)])
"""))

cells.append(md("## 12) Metrik yardımcıları"))
cells.append(code("""
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_reg(reg, Xd, yd):
    pred = reg.predict(Xd)
    return {
        "rmse": rmse(yd, pred),
        "mae": float(mean_absolute_error(yd, pred)),
        "r2": float(r2_score(yd, pred)),
    }

# Naif baseline: train ortalamasi
naive_pred = np.full_like(y_val.values, y_train.mean(), dtype=float)
print("Naif (ortalama) baseline -> val RMSE:", round(rmse(y_val, naive_pred), 3),
      "| MAE:", round(mean_absolute_error(y_val, naive_pred), 3),
      "| R2:", round(r2_score(y_val, naive_pred), 3))
"""))

cells.append(md("## 13) Çoklu model karşılaştırması (validation)"))
cells.append(code("""
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=300),
}

rows = []
for name, m in models.items():
    pipe = build_reg(m)
    pipe.fit(X_train_fe, y_train)
    tr = evaluate_reg(pipe, X_train_fe, y_train)
    vl = evaluate_reg(pipe, X_val_fe, y_val)
    rows.append({
        "model": name,
        "train_rmse": tr["rmse"], "val_rmse": vl["rmse"],
        "train_mae": tr["mae"], "val_mae": vl["mae"],
        "train_r2": tr["r2"], "val_r2": vl["r2"],
    })

baseline_df = pd.DataFrame(rows).sort_values("val_rmse")
baseline_df
"""))

cells.append(md("## 14) Cross-validation (train+val)"))
cells.append(code("""
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
X_full = pd.concat([X_train_fe, X_val_fe], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)

best_row = baseline_df.iloc[0]
best_base_name = best_row["model"]
base_estimator = models[best_base_name]
cv_pipe = build_reg(clone(base_estimator))

rmse_cv = cross_val_score(cv_pipe, X_full, y_full, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
r2_cv = cross_val_score(cv_pipe, X_full, y_full, cv=cv, scoring="r2", n_jobs=1)
print("CV secilen model:", best_base_name)
print("CV RMSE mean/std:", round(-rmse_cv.mean(), 4), round(rmse_cv.std(), 4))
print("CV R2 mean/std:", round(r2_cv.mean(), 4), round(r2_cv.std(), 4))
"""))

cells.append(md("## 15) Hiperparametre tuning (RandomizedSearchCV)"))
cells.append(code("""
param_spaces = {
    "LinearRegression": {},
    "Ridge": {"model__alpha": np.logspace(-3, 3, 14)},
    "Lasso": {"model__alpha": np.logspace(-4, 1, 14)},
    "RandomForest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 8, 14, 22],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.6, 1.0],
    },
    "ExtraTrees": {
        "model__n_estimators": [300, 500, 700],
        "model__max_depth": [None, 10, 18, 28],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "model__n_estimators": [150, 250, 400],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__subsample": [0.7, 0.85, 1.0],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_depth": [None, 6, 10, 16],
        "model__max_leaf_nodes": [31, 63, 127],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}

space = param_spaces.get(best_base_name, param_spaces["HistGradientBoosting"])
if space:
    tuned = RandomizedSearchCV(
        build_reg(clone(base_estimator)),
        space,
        n_iter=20,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    tuned.fit(X_full, y_full)
    print("En iyi CV RMSE:", round(-tuned.best_score_, 4))
    print("Parametreler:", tuned.best_params_)
    best_model = tuned.best_estimator_
else:
    print("Aranacak parametre uzayi yok; baseline modeli kullaniyoruz.")
    best_model = build_reg(clone(base_estimator))
    best_model.fit(X_full, y_full)
"""))

cells.append(md("## 16) Final test değerlendirmesi"))
cells.append(code("""
best_model.fit(X_full, y_full)
test_metrics = evaluate_reg(best_model, X_test_fe, y_test)
print("Test metrikleri:", {k: round(v, 4) for k, v in test_metrics.items()})

y_pred_test = best_model.predict(X_test_fe)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_test, y_pred_test, alpha=0.35, s=18)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0].set_xlabel("Gercek Recovery_Score")
axes[0].set_ylabel("Tahmin")
axes[0].set_title(f"Gercek vs Tahmin (R^2 = {test_metrics['r2']:.3f})")

residuals = y_test.values - y_pred_test
sns.histplot(residuals, kde=True, ax=axes[1], color="#C44E52")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Artik (gercek - tahmin)")
axes[1].set_title(f"Artik dagilimi (mean={residuals.mean():.2f}, std={residuals.std():.2f})")
plt.tight_layout()
plt.show()
"""))

cells.append(md("## 17) Özellik önemi (permutation / built-in)"))
cells.append(code("""
from sklearn.inspection import permutation_importance

try:
    final_estimator = best_model.named_steps["model"]
    pre_step = best_model.named_steps["pre"]
    feat_names_num = num_features
    try:
        ohe = pre_step.named_transformers_["cat"].named_steps["oh"]
        feat_names_cat = list(ohe.get_feature_names_out(cat_features))
    except Exception:
        feat_names_cat = []
    feat_names = feat_names_num + feat_names_cat
    importances = getattr(final_estimator, "feature_importances_", None)
    if importances is not None and len(importances) == len(feat_names):
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(8, 7))
        sns.barplot(data=imp_df, y="feature", x="importance", color="#4C72B0")
        plt.title("Model feature_importances_ (top 20)")
        plt.tight_layout()
        plt.show()
        print(imp_df.round(4).to_string(index=False))
    else:
        print("feature_importances_ yok; permutation importance ile gosteriliyor (kucuk ornek).")
        sample = X_test_fe.sample(min(800, len(X_test_fe)), random_state=RANDOM_STATE)
        sample_y = y_test.loc[sample.index]
        perm = permutation_importance(best_model, sample, sample_y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1, scoring="r2")
        perm_df = pd.DataFrame({
            "feature": X_test_fe.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=perm_df, y="feature", x="importance_mean", color="#55A868")
        plt.title("Permutation importance (R^2 dususu)")
        plt.tight_layout()
        plt.show()
        print(perm_df.round(4).to_string(index=False))
except Exception as e:
    print("Feature importance hesaplanamadi:", e)
"""))

cells.append(md("## 18) Hata analizi (en kötü tahminler)"))
cells.append(code("""
err = X_test_fe.copy()
err["y_true"] = y_test.values
err["y_pred"] = np.round(y_pred_test, 2)
err["abs_error"] = np.round(np.abs(err["y_true"] - err["y_pred"]), 2)
err.sort_values("abs_error", ascending=False).head(12)
"""))

cells.append(md("## 19) Model kaydetme + örnek tahmin"))
cells.append(code("""
os.makedirs("models", exist_ok=True)
bundle = {
    "pipeline": best_model,
    "target_column": TARGET,
    "feature_columns": list(X_train_fe.columns),
    "split_mode": SPLIT_MODE,
}
path_model = "models/athlete_recovery_model.joblib"
joblib.dump(bundle, path_model)
print("Kaydedildi:", path_model)

loaded = joblib.load(path_model)
pipe = loaded["pipeline"]
sample = X_test_fe.head(8).copy()
preds = pipe.predict(sample)
out = sample[[c for c in ["Training_Type", "Training_Intensity", "Sleep_Duration_Hours", "HRV_ms", "Resting_Heart_Rate", "Stress_Level", "Mood_Score"] if c in sample.columns]].copy()
out["gercek_Recovery"] = y_test.iloc[:8].values
out["tahmin_Recovery"] = np.round(preds, 1)
out
"""))

cells.append(md("""
## 20) Sonuç

- 300 atletin 28 günlük antrenman & biyometri verisinden **Recovery_Score** sürekli değişkenini tahmin eden bir regresyon pipeline'ı kuruldu.
- Önemli adımlar: eksik veri imputasyonu (özellikle `Sleep_Duration_Hours`), one-hot kodlama, **Training_Load**, **Sleep_Deficit**, **HRV_RHR_Ratio** gibi türetilmiş özellikler.
- Atlet bazlı split (GroupShuffleSplit) sayesinde aynı atletin günleri train/test arasında karışmıyor; bu daha gerçekçi bir genelleme ölçümü verir.
- En iyi model **RandomizedSearchCV** ile ayarlandı; test setinde RMSE / MAE / R² raporlandı, artık (residual) analizi ve özellik önemi çıkarıldı.

**Sonraki adımlar:**
- Atlet bazlı **rolling/lag** özellikler (önceki günün uykusu, son 3 günün HRV ortalaması),
- Kişisel tahmin için **mixed effects** veya atlet sabit etki modelleri,
- LightGBM / XGBoost ile karşılaştırma, SHAP ile yorumlanabilirlik,
- Recovery_Score'u **Poor / Moderate / Excellent** sınıflarına çevirip sınıflandırma versiyonu.

**Kaggle:** [Athlete Recovery & Biometric Performance Dataset](https://www.kaggle.com/datasets/sarveshchhetri/athlete-recovery-and-biometric-performance-dataset)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Notebook yazildi:", NB_PATH, "| Hucre:", len(cells))
