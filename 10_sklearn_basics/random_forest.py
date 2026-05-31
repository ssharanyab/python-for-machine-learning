# %% [markdown]
# # Random Forest
#
# ## Why this matters
# Random Forest is one of the most reliable and widely-used ML algorithms. It conquers the
# overfitting problem of single decision trees through **bagging** (bootstrap aggregating) and
# **random feature subsets**. It delivers strong out-of-the-box performance, built-in
# feature importance, and OOB error estimation.
#
# ## Learning Objectives
# - Understand bagging and why it reduces variance
# - Train Random Forest classifiers and regressors
# - Tune key hyperparameters: `n_estimators`, `max_depth`, `max_features`
# - Use OOB score and feature importances

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

# %% [markdown]
# ## Concept Explanation
# Random Forest builds `n_estimators` decision trees, each trained on a **bootstrap sample**
# of the data with a **random subset of features** at each split.
#
# - **Bagging**: each tree sees ~63% of data (bootstrap); remaining ~37% is OOB
# - **Feature randomness**: `max_features` features considered per split (default: sqrt(p))
# - **Aggregation**: majority vote (classification) or mean (regression)
#
# This diversity between trees reduces variance dramatically compared to a single tree.

# %% [markdown]
# ## Beginner Examples

# %%
# 1. Random Forest vs single Decision Tree — breast cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

dt.fit(X_tr, y_tr)
rf.fit(X_tr, y_tr)

print(f"Decision Tree  — Train: {dt.score(X_tr, y_tr):.4f} | Test: {dt.score(X_te, y_te):.4f}")
print(f"Random Forest  — Train: {rf.score(X_tr, y_tr):.4f} | Test: {rf.score(X_te, y_te):.4f}")

# %%
# 2. Out-of-Bag error (free validation without a held-out set)
rf_oob = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf_oob.fit(X_tr, y_tr)
print(f"OOB Score  : {rf_oob.oob_score_:.4f}")
print(f"Test Score : {rf_oob.score(X_te, y_te):.4f}")

# %%
# 3. Feature importances
importances = pd.Series(rf.feature_importances_, index=cancer.feature_names)
top10 = importances.nlargest(10).sort_values()
plt.figure(figsize=(9, 5))
top10.plot(kind="barh", color="#4C72B0", edgecolor="white")
plt.title("Top-10 Feature Importances — Random Forest (Breast Cancer)", fontsize=13, fontweight="bold")
plt.xlabel("Importance", fontsize=12); plt.grid(axis="x", alpha=0.4)
plt.tight_layout(); plt.show()

# %%
# 4. Effect of n_estimators
n_trees = [1, 5, 10, 25, 50, 100, 200, 500]
test_accs = [RandomForestClassifier(n_estimators=n, random_state=42).fit(X_tr, y_tr).score(X_te, y_te)
             for n in n_trees]

plt.figure(figsize=(8, 4))
plt.plot(n_trees, test_accs, "o-", color="#4C72B0", linewidth=2)
plt.title("Number of Trees vs. Test Accuracy", fontsize=13, fontweight="bold")
plt.xlabel("n_estimators", fontsize=12); plt.ylabel("Accuracy", fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %%
# 5. Classification report
print(classification_report(y_te, rf.predict(X_te), target_names=cancer.target_names))

# %% [markdown]
# ## Intermediate Examples

# %%
# 6. Random Forest Regressor — California Housing
housing = fetch_california_housing()
Xh, yh = housing.data, housing.target
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)

rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(Xh_tr, yh_tr)
rmse = np.sqrt(mean_squared_error(yh_te, rfr.predict(Xh_te)))
print(f"Random Forest Regressor RMSE: {rmse:.4f}")
print(f"Random Forest Regressor R²  : {rfr.score(Xh_te, yh_te):.4f}")

# %%
# 7. RandomizedSearchCV for hyperparameter tuning
param_dist = {
    "n_estimators":  [50, 100, 200],
    "max_depth":     [None, 5, 10, 20],
    "max_features":  ["sqrt", "log2", 0.5],
    "min_samples_split": [2, 5, 10],
}
rscv = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), param_dist,
    n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=-1
)
rscv.fit(X_tr, y_tr)
print("Best params:", rscv.best_params_)
print(f"Best CV accuracy: {rscv.best_score_:.4f}")
print(f"Test accuracy   : {rscv.score(X_te, y_te):.4f}")

# %%
# 8. Confidence from vote counts
rf100 = RandomForestClassifier(n_estimators=100, random_state=42)
rf100.fit(X_tr, y_tr)
proba = rf100.predict_proba(X_te[:5])
for i, p in enumerate(proba):
    print(f"Sample {i}: P(malignant)={p[0]:.3f}, P(benign)={p[1]:.3f}")

# %% [markdown]
# ## Machine Learning Relevance
# - Random Forest is a strong **general-purpose baseline** for tabular data
# - **OOB score** provides a free estimate of generalisation without a validation split
# - **Feature importances** drive feature selection in production pipelines
# - Works well with **missing data** (via surrogate splits) and **high-dimensional** inputs

# %% [markdown]
# ## Common Mistakes
# 1. **Using too few trees** — accuracy is unstable; use ≥100
# 2. **Not setting `n_jobs=-1`** on large datasets — wastes compute
# 3. **Treating feature importances as absolute** — they can be biased toward high-cardinality features
# 4. **Ignoring `max_depth`** — unlimited trees still overfit on noisy data
# 5. **Not using `oob_score=True`** — free validation check you should always enable

# %% [markdown]
# ## Interview Questions
# 1. How does bagging reduce variance without increasing bias?
# 2. What is the OOB error and why is it a reliable estimate of generalisation?
# 3. What is the difference between `max_features='sqrt'` and `max_features=1.0`?
# 4. Why does Random Forest work well without feature scaling?
# 5. How does Random Forest handle missing values, and what are its limitations?

# %% [markdown]
# ## Practice Problems
# 1. Compare Random Forest with 50, 100, and 300 trees on `load_digits()` using 5-fold CV.
# 2. Extract and plot permutation importances on the test set and compare to `.feature_importances_`.
# 3. Implement a manual bagging classifier using 10 Decision Trees and majority vote.
# 4. Use `RandomizedSearchCV` to tune Random Forest on the housing dataset; report final RMSE.
# 5. Plot the distribution of OOB scores across 10 different `random_state` values.

# %% [markdown]
# ## Solutions

# %%
# Problem 1
from sklearn.datasets import load_digits
digits = load_digits()
for n in [50, 100, 300]:
    s = cross_val_score(RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1),
                        digits.data, digits.target, cv=5).mean()
    print(f"n_estimators={n:3d} → CV accuracy={s:.4f}")

# %%
# Problem 3 — manual bagging
from sklearn.utils import resample
from scipy.stats import mode

np.random.seed(42)
trees = []
for _ in range(10):
    Xb, yb = resample(X_tr, y_tr, random_state=_)
    t = DecisionTreeClassifier(random_state=_)
    t.fit(Xb, yb)
    trees.append(t)

preds = np.array([t.predict(X_te) for t in trees])
majority = mode(preds, axis=0).mode.ravel()
print(f"Manual bagging test accuracy: {accuracy_score(y_te, majority):.4f}")

# %%
from sklearn.metrics import accuracy_score  # ensure imported for above

# %% [markdown]
# ## Further Reading
# - [sklearn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
# - [Breiman (2001) — Random Forests paper](https://link.springer.com/article/10.1023/A:1010933404324)
# - Géron, *Hands-On ML*, Ch. 7 — Ensemble Learning
