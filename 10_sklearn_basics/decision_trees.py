# %% [markdown]
# # Decision Trees
#
# ## Why this matters
# Decision trees are intuitive, interpretable, and form the foundation of powerful ensemble
# methods (Random Forest, Gradient Boosting, XGBoost). Understanding how they work — splits,
# impurity, depth — is essential for any ML practitioner.
#
# ## Learning Objectives
# - Understand how decision trees split data using Gini impurity and entropy
# - Train classification and regression trees with sklearn
# - Visualise and interpret a decision tree
# - Tune max_depth to prevent overfitting

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance

# %% [markdown]
# ## Concept Explanation
# A decision tree partitions the feature space into rectangular regions by asking a series
# of yes/no questions (splits). Each internal node represents a feature test; each leaf
# holds a prediction.
#
# **Splitting criteria:**
# - **Gini impurity**: `G = 1 - Σ pᵢ²` (default for classification)
# - **Entropy**: `H = -Σ pᵢ log₂(pᵢ)` (information gain)
# - **MSE**: used for regression trees

# %% [markdown]
# ## Beginner Examples

# %%
# 1. Train a classification tree on Iris
iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_tr, y_tr)
print(f"Train accuracy: {dt.score(X_tr, y_tr):.4f}")
print(f"Test  accuracy: {dt.score(X_te, y_te):.4f}")

# %%
# 2. Visualise the tree
fig, ax = plt.subplots(figsize=(16, 6))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, ax=ax, fontsize=9)
ax.set_title("Decision Tree — Iris Dataset (max_depth=3)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# %%
# 3. Text representation
print(export_text(dt, feature_names=list(iris.feature_names)))

# %%
# 4. Feature importance
importances = pd.Series(dt.feature_importances_, index=iris.feature_names).sort_values(ascending=True)
plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="#4C72B0", edgecolor="white")
plt.title("Feature Importances — Iris Decision Tree", fontsize=13, fontweight="bold")
plt.xlabel("Importance (Gini)", fontsize=12); plt.grid(axis="x", alpha=0.4)
plt.tight_layout(); plt.show()

# %%
# 5. Predict on new data
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # likely setosa
pred = dt.predict(sample)
prob = dt.predict_proba(sample)
print(f"Predicted class: {iris.target_names[pred[0]]}")
print(f"Class probabilities: {dict(zip(iris.target_names, prob[0].round(3)))}")

# %% [markdown]
# ## Intermediate Examples

# %%
# 6. Overfitting: depth vs accuracy
cancer = load_breast_cancer()
Xc, yc = cancer.data, cancer.target
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

depths = range(1, 20)
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(Xc_tr, yc_tr)
    train_accs.append(m.score(Xc_tr, yc_tr))
    test_accs.append(m.score(Xc_te, yc_te))

plt.figure(figsize=(9, 4))
plt.plot(depths, train_accs, "o-", label="Train", color="#4C72B0")
plt.plot(depths, test_accs, "s-", label="Test",  color="#DD8452")
plt.title("Decision Tree Depth vs. Accuracy", fontsize=13, fontweight="bold")
plt.xlabel("max_depth", fontsize=12); plt.ylabel("Accuracy", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %%
# 7. Regression tree
housing = fetch_california_housing()
Xh, yh = housing.data, housing.target
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)

dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
dtr.fit(Xh_tr, yh_tr)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(yh_te, dtr.predict(Xh_te)))
print(f"Regression Tree RMSE: {rmse:.4f}")
print(f"Regression Tree R²  : {dtr.score(Xh_te, yh_te):.4f}")

# %%
# 8. Cross-validation for robust evaluation
cv = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=42), Xc, yc, cv=5)
print(f"5-Fold CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# %% [markdown]
# ## Machine Learning Relevance
# - Decision trees are the base estimator for **Random Forest** and **Gradient Boosting**
# - Feature importances are used for feature selection
# - Trees handle mixed feature types natively (no scaling needed)
# - `max_depth`, `min_samples_split`, `min_samples_leaf` are key regularisation knobs

# %% [markdown]
# ## Common Mistakes
# 1. **Unlimited depth** — the tree memorises training data (overfits)
# 2. **Not using cross-validation** — single split accuracy is noisy
# 3. **Ignoring class imbalance** — use `class_weight='balanced'`
# 4. **Treating feature importance as causation** — it measures predictive power, not causality
# 5. **Using a single tree for production** — prefer ensembles (Random Forest)

# %% [markdown]
# ## Interview Questions
# 1. What is Gini impurity and how is it used to choose a split in a decision tree?
# 2. Why do decision trees overfit easily, and what hyperparameters control this?
# 3. What is the difference between a classification tree and a regression tree?
# 4. How are feature importances calculated in a decision tree?
# 5. Why are decision trees considered "white-box" models? What are the trade-offs?

# %% [markdown]
# ## Practice Problems
# 1. Train a decision tree on `load_digits()` with `max_depth=None`. Observe overfitting.
# 2. Find the optimal `max_depth` for the cancer dataset using 5-fold CV and a depth range of 1-15.
# 3. Prune the tree using `ccp_alpha` (cost-complexity pruning). Plot accuracy vs. alpha.
# 4. Use `permutation_importance` to rank features on the test set. Compare with `.feature_importances_`.
# 5. Build a regression tree on housing data with `max_leaf_nodes=20` and visualise predictions vs. actual.

# %% [markdown]
# ## Solutions

# %%
# Problem 1
from sklearn.datasets import load_digits
digits = load_digits()
dt_deep = DecisionTreeClassifier(max_depth=None, random_state=42)
Xdig_tr, Xdig_te, ydig_tr, ydig_te = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42)
dt_deep.fit(Xdig_tr, ydig_tr)
print(f"Train: {dt_deep.score(Xdig_tr, ydig_tr):.4f} | Test: {dt_deep.score(Xdig_te, ydig_te):.4f}")
print("Tree depth:", dt_deep.get_depth(), "| Leaves:", dt_deep.get_n_leaves())

# %%
# Problem 2
best_d, best_cv = 1, 0
for d in range(1, 16):
    s = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), Xc, yc, cv=5).mean()
    if s > best_cv:
        best_cv, best_d = s, d
print(f"Best max_depth={best_d} with CV accuracy={best_cv:.4f}")

# %%
# Problem 3 — cost-complexity pruning
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(Xc_tr, yc_tr)
alphas = path.ccp_alphas
accs = [DecisionTreeClassifier(ccp_alpha=a, random_state=42).fit(Xc_tr, yc_tr).score(Xc_te, yc_te)
        for a in alphas]
plt.figure(figsize=(8, 4))
plt.plot(alphas, accs, "o-", color="#4C72B0")
plt.title("CCP Alpha vs Test Accuracy", fontsize=13, fontweight="bold")
plt.xlabel("ccp_alpha"); plt.ylabel("Accuracy"); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Further Reading
# - [sklearn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
# - [sklearn cost-complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
# - Géron, *Hands-On ML*, Ch. 6 — Decision Trees
