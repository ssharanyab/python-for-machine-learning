# %% [markdown]
# # Model Evaluation
#
# ## Why this matters
# Choosing the right metric is as important as choosing the right model. Accuracy alone is
# misleading on imbalanced data. A strong ML engineer understands precision, recall, F1,
# ROC-AUC, confusion matrices, and regression metrics — and knows when to use each.
#
# ## Learning Objectives
# - Compute and interpret accuracy, precision, recall, F1, and ROC-AUC
# - Build and read confusion matrices
# - Apply regression metrics: MAE, MSE, RMSE, R²
# - Diagnose models using learning curves and calibration plots

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, average_precision_score
)

# %% [markdown]
# ## Concept Explanation
#
# ### Classification Metrics
# | Metric | Formula | When to use |
# |---|---|---|
# | Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
# | Precision | TP/(TP+FP) | Cost of false positives is high |
# | Recall | TP/(TP+FN) | Cost of false negatives is high |
# | F1 | 2·P·R/(P+R) | Balance precision & recall |
# | ROC-AUC | Area under ROC curve | Ranking quality across thresholds |
#
# ### Regression Metrics
# - **MAE**: Mean Absolute Error — robust to outliers
# - **MSE/RMSE**: Penalises large errors more
# - **R²**: Proportion of variance explained (1.0 = perfect)

# %% [markdown]
# ## Beginner Examples

# %%
# 1. Setup: train a classifier
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
y_proba = rf.predict_proba(X_te)[:, 1]

# %%
# 2. Core classification metrics
print("=== Classification Metrics ===")
print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall   : {recall_score(y_te, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_te, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_te, y_proba):.4f}")

# %%
# 3. Full classification report
print(classification_report(y_te, y_pred, target_names=cancer.target_names))

# %%
# 4. Confusion matrix
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Random Forest (Breast Cancer)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nTP={tp} | TN={tn} | FP={fp} | FN={fn}")
print(f"Specificity (True Negative Rate): {tn/(tn+fp):.4f}")

# %%
# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_te, y_proba)
auc = roc_auc_score(y_te, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {auc:.3f}")
plt.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
plt.plot([0,1],[0,1],"k--", alpha=0.5, label="Random")
plt.title("ROC Curve", fontsize=14, fontweight="bold")
plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Intermediate Examples

# %%
# 6. Precision-Recall Curve (better for imbalanced data)
precision_arr, recall_arr, _ = precision_recall_curve(y_te, y_proba)
ap = average_precision_score(y_te, y_proba)

plt.figure(figsize=(7, 5))
plt.plot(recall_arr, precision_arr, color="#DD8452", linewidth=2, label=f"AP = {ap:.3f}")
plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
plt.xlabel("Recall", fontsize=12); plt.ylabel("Precision", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %%
# 7. Compare multiple models
models = {
    "Logistic Reg": Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
results = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    ypr = m.predict_proba(X_te)[:, 1]
    results[name] = {
        "Accuracy": accuracy_score(y_te, yp),
        "Precision": precision_score(y_te, yp),
        "Recall": recall_score(y_te, yp),
        "F1": f1_score(y_te, yp),
        "ROC-AUC": roc_auc_score(y_te, ypr),
    }

df_results = pd.DataFrame(results).T
print(df_results.round(4))

# %%
# 8. Visualise model comparison
df_results.plot(kind="bar", figsize=(10, 5), colormap="viridis", edgecolor="white")
plt.title("Model Comparison — Classification Metrics", fontsize=14, fontweight="bold")
plt.xlabel("Model", fontsize=12); plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=0); plt.legend(loc="lower right"); plt.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()

# %%
# 9. Regression metrics
housing = fetch_california_housing()
Xh, yh = housing.data, housing.target
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)
rfr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfr.fit(Xh_tr, yh_tr)
yh_pred = rfr.predict(Xh_te)

print("=== Regression Metrics ===")
print(f"MAE  : {mean_absolute_error(yh_te, yh_pred):.4f}")
print(f"MSE  : {mean_squared_error(yh_te, yh_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(yh_te, yh_pred)):.4f}")
print(f"R²   : {r2_score(yh_te, yh_pred):.4f}")

# %%
# 10. Learning curve — bias/variance diagnosis
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy", n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
test_mean  = test_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
test_std   = test_scores.std(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Train")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color="#4C72B0")
plt.plot(train_sizes, test_mean, "s-", color="#DD8452", label="CV")
plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.15, color="#DD8452")
plt.title("Learning Curve — Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Training Size", fontsize=12); plt.ylabel("Accuracy", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Machine Learning Relevance
# - In medical ML, **recall** (sensitivity) matters more than precision — missing a disease is costly
# - For spam detection, **precision** matters — false positives are annoying
# - **ROC-AUC** is threshold-independent — use it when comparing models, not deploying them
# - **Learning curves** directly diagnose underfitting (high bias) vs. overfitting (high variance)

# %% [markdown]
# ## Common Mistakes
# 1. **Using accuracy on imbalanced data** — a 99% negative class gives 99% accuracy with a dummy model
# 2. **Optimising for F1 when you need recall** — know your business cost function
# 3. **Not reporting confidence intervals** — single-split metrics are noisy
# 4. **Evaluating on training data** — always use held-out or CV scores
# 5. **Ignoring calibration** — high AUC doesn't mean well-calibrated probabilities

# %% [markdown]
# ## Interview Questions
# 1. Explain precision and recall. Give a real-world example where you'd prioritise each.
# 2. What is the F1 score and why is it the harmonic mean (not arithmetic)?
# 3. When is ROC-AUC misleading? What alternative would you use for imbalanced data?
# 4. What does a learning curve tell you about bias and variance?
# 5. What is the difference between macro, micro, and weighted averaging in multi-class F1?

# %% [markdown]
# ## Practice Problems
# 1. Create an imbalanced binary dataset (95/5 split) and show that accuracy is misleading. Use F1 instead.
# 2. For the cancer classifier, find the threshold that maximises F1 score.
# 3. Plot a calibration curve using `CalibrationDisplay` and interpret the result.
# 4. Compute the Matthews Correlation Coefficient (MCC) and explain why it is more informative than F1 on imbalanced data.
# 5. Build a regression model and plot actual vs. predicted values, colouring points by residual magnitude.

# %% [markdown]
# ## Solutions

# %%
# Problem 1 — imbalanced accuracy trap
from sklearn.datasets import make_classification
X_imb, y_imb = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
Xib_tr, Xib_te, yib_tr, yib_te = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)
dummy_preds = np.zeros_like(yib_te)  # always predict majority class
print(f"Dummy accuracy : {accuracy_score(yib_te, dummy_preds):.4f}")
print(f"Dummy F1 (minority): {f1_score(yib_te, dummy_preds, zero_division=0):.4f}")

rf_imb = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_imb.fit(Xib_tr, yib_tr)
print(f"RF accuracy : {accuracy_score(yib_te, rf_imb.predict(Xib_te)):.4f}")
print(f"RF F1 (minority): {f1_score(yib_te, rf_imb.predict(Xib_te)):.4f}")

# %%
# Problem 2 — optimal F1 threshold
y_proba2 = rf.predict_proba(X_te)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = [f1_score(y_te, (y_proba2 >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
print(f"Best threshold: {best_t:.2f} → F1={max(f1s):.4f}")

# %%
# Problem 4 — MCC
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_te, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# %% [markdown]
# ## Further Reading
# - [sklearn metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
# - [Google ML Crash Course — Classification](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
# - Géron, *Hands-On ML*, Ch. 3 — Classification
