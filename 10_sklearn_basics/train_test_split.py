# %% [markdown]
# # Train-Test Split
#
# ## Why this matters
# Splitting data into training and test sets is the cornerstone of any honest ML evaluation.
# Without a held-out test set you cannot detect overfitting, and your model's reported
# performance will not generalise to real-world data.
#
# ## Learning Objectives
# - Understand why we split data and what problems it prevents
# - Use `train_test_split` correctly with stratification and random seeds
# - Understand validation sets and cross-validation
# - Avoid data leakage pitfalls

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# %% [markdown]
# ## Concept Explanation
# The fundamental goal: estimate how well a model generalises to **unseen** data.
#
# - **Training set** — the model learns from this
# - **Validation set** — used to tune hyperparameters (optional but recommended)
# - **Test set** — touched ONCE at the very end for final evaluation
#
# A typical split is 80/20 or 70/30 train/test.

# %% [markdown]
# ## Beginner Examples

# %%
# 1. Basic split on the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# %%
# 2. Verify class distribution is maintained with stratify
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
unique, counts = np.unique(y_test_s, return_counts=True)
print("Test class distribution (stratified):", dict(zip(iris.target_names, counts)))

# %%
# 3. Split a pandas DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# %%
# 4. Effect of different random seeds
for seed in [0, 1, 42, 99]:
    _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    acc = model.score(X_t, y_t)
    print(f"Seed {seed:2d} → Test accuracy: {acc:.4f}")

# %%
# 5. Visualise the split sizes
sizes = [X_train.shape[0], X_test.shape[0]]
labels = ["Train (80%)", "Test (20%)"]
colors = ["#4C72B0", "#DD8452"]
plt.figure(figsize=(6, 4))
plt.bar(labels, sizes, color=colors, edgecolor="white", linewidth=1.5)
plt.title("Dataset Split Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Split", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.grid(axis="y", alpha=0.4)
for i, v in enumerate(sizes):
    plt.text(i, v + 1, str(v), ha="center", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Intermediate Examples

# %%
# 6. K-Fold Cross-Validation for more robust estimates
cancer = load_breast_cancer()
Xc, yc = cancer.data, cancer.target
model = LogisticRegression(max_iter=5000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, Xc, yc, cv=kf, scoring="accuracy")
print(f"5-Fold CV Scores: {cv_scores.round(4)}")
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %%
# 7. Stratified K-Fold (recommended for imbalanced classes)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_s = cross_val_score(model, Xc, yc, cv=skf, scoring="f1")
print(f"Stratified CV F1: {cv_scores_s.mean():.4f} ± {cv_scores_s.std():.4f}")

# %%
# 8. Train vs test accuracy curve (bias-variance tradeoff demo)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

X_1d = np.sort(np.random.RandomState(42).rand(100, 1) * 10, axis=0)
y_1d = np.sin(X_1d).ravel() + np.random.RandomState(42).randn(100) * 0.3
X_tr, X_te, y_tr, y_te = train_test_split(X_1d, y_1d, test_size=0.2, random_state=42)

degrees = range(1, 12)
train_errs, test_errs = [], []
for d in degrees:
    pipe = Pipeline([("poly", PolynomialFeatures(d)), ("ridge", Ridge())])
    pipe.fit(X_tr, y_tr)
    train_errs.append(np.mean((pipe.predict(X_tr) - y_tr) ** 2))
    test_errs.append(np.mean((pipe.predict(X_te) - y_te) ** 2))

plt.figure(figsize=(8, 4))
plt.plot(degrees, train_errs, "o-", label="Train MSE", color="#4C72B0")
plt.plot(degrees, test_errs, "s-", label="Test MSE", color="#DD8452")
plt.xlabel("Polynomial Degree", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.title("Bias-Variance Tradeoff via Train/Test Split", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Machine Learning Relevance
# - **Data leakage**: never fit scalers/encoders on the full dataset before splitting
# - **Stratification** is critical for imbalanced classification tasks
# - **Cross-validation** gives more reliable estimates on small datasets
# - Reproducibility requires a fixed `random_state`

# %%
# Correct pipeline — fit scaler only on training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit+transform on train
X_test_scaled  = scaler.transform(X_test)         # transform only on test
print("Scaler fitted on train only — no data leakage ✓")

# %% [markdown]
# ## Common Mistakes
# 1. **Fitting the scaler on the full dataset** before splitting → data leakage
# 2. **Not using `stratify`** on imbalanced datasets → misleading accuracy
# 3. **Evaluating repeatedly on the test set** and then tuning → test set becomes validation set
# 4. **Ignoring `random_state`** → non-reproducible results
# 5. **Using too small a test set** on small datasets → use cross-validation instead

# %% [markdown]
# ## Interview Questions
# 1. What is data leakage and how does train-test split prevent it?
# 2. When should you use stratified splitting versus regular splitting?
# 3. What is the difference between a validation set and a test set?
# 4. Why might K-Fold cross-validation give a better estimate than a single train-test split?
# 5. How does `random_state` affect reproducibility in ML experiments?

# %% [markdown]
# ## Practice Problems
# 1. Load the `load_digits()` dataset and split it 70/30 with stratification. Report class balance in both splits.
# 2. Implement 10-fold cross-validation on the breast cancer dataset and plot the fold-by-fold accuracy.
# 3. Demonstrate data leakage: fit a `StandardScaler` on the full dataset vs. train only, and compare the test accuracy.
# 4. Write a function `evaluate_splits(X, y, test_sizes)` that returns mean CV accuracy for each test size.
# 5. Using the housing dataset, perform a 5-fold CV with Ridge regression and report RMSE per fold.

# %% [markdown]
# ## Solutions

# %%
# Problem 1
from sklearn.datasets import load_digits
digits = load_digits()
Xd, yd = digits.data, digits.target
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, random_state=42, stratify=yd)
print("Train class counts:", np.bincount(yd_tr))
print("Test class counts :", np.bincount(yd_te))

# %%
# Problem 2
cv10 = cross_val_score(LogisticRegression(max_iter=5000), Xc, yc, cv=10, scoring="accuracy")
plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), cv10, color="#4C72B0", edgecolor="white")
plt.axhline(cv10.mean(), color="#DD8452", linestyle="--", label=f"Mean={cv10.mean():.3f}")
plt.xlabel("Fold", fontsize=12); plt.ylabel("Accuracy", fontsize=12)
plt.title("10-Fold CV Accuracy — Breast Cancer Dataset", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Further Reading
# - [sklearn train_test_split docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
# - [sklearn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
# - Géron, *Hands-On ML*, Ch. 2 — End-to-End ML Project
