# %% [markdown]
# # Logistic Regression
#
# ## Why this matters
# Despite its name, logistic regression is a **classification** algorithm — and one of the
# most important in ML. It is the building block of binary classifiers, outputs calibrated
# probabilities, and is directly interpretable. It is also the foundation of neural network
# output layers.
#
# ## Learning Objectives
# - Understand the sigmoid function and log-odds
# - Train binary and multi-class logistic regression with sklearn
# - Interpret coefficients and predict probabilities
# - Evaluate with ROC-AUC and classification reports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## Concept Explanation
# Logistic regression models the **probability** that a sample belongs to class 1:
#
# `P(y=1 | X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)` where `σ(z) = 1 / (1 + e⁻ᶻ)` (sigmoid)
#
# Training minimises **binary cross-entropy loss**:
#
# `L = -1/n Σ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]`

# %% [markdown]
# ## Beginner Examples

# %%
# 1. Visualise the sigmoid function
z = np.linspace(-8, 8, 200)
sigmoid = 1 / (1 + np.exp(-z))
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid, color="#4C72B0", linewidth=2.5)
plt.axhline(0.5, color="#DD8452", linestyle="--", alpha=0.7, label="Decision boundary (p=0.5)")
plt.title("Sigmoid (Logistic) Function", fontsize=14, fontweight="bold")
plt.xlabel("z (log-odds)", fontsize=12); plt.ylabel("P(y=1)", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %%
# 2. Binary classification — breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
pipe.fit(X_tr, y_tr)
print(f"Train accuracy: {pipe.score(X_tr, y_tr):.4f}")
print(f"Test  accuracy: {pipe.score(X_te, y_te):.4f}")

# %%
# 3. Probability predictions
probs = pipe.predict_proba(X_te[:5])
print("Class probabilities (first 5 samples):")
for i, p in enumerate(probs):
    print(f"  Sample {i}: P(malignant)={p[0]:.3f}, P(benign)={p[1]:.3f}")

# %%
# 4. Classification report
y_pred = pipe.predict(X_te)
print(classification_report(y_te, y_pred, target_names=cancer.target_names))

# %%
# 5. Confusion matrix
cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Breast Cancer", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Intermediate Examples

# %%
# 6. ROC Curve and AUC
y_scores = pipe.predict_proba(X_te)[:, 1]
fpr, tpr, thresholds = roc_curve(y_te, y_scores)
auc = roc_auc_score(y_te, y_scores)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
plt.title("ROC Curve — Breast Cancer Classifier", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %%
# 7. Multi-class (Iris dataset)
iris = load_iris()
Xi, yi = iris.data, iris.target
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.2, random_state=42, stratify=yi)

lr_multi = Pipeline([("scaler", StandardScaler()),
                      ("lr", LogisticRegression(multi_class="multinomial", max_iter=500))])
lr_multi.fit(Xi_tr, yi_tr)
print(f"Iris multi-class accuracy: {lr_multi.score(Xi_te, yi_te):.4f}")
print(classification_report(yi_te, lr_multi.predict(Xi_te), target_names=iris.target_names))

# %%
# 8. Regularisation: C parameter (inverse of lambda)
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
cv_scores = []
for C in C_values:
    p = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=2000))])
    cv_scores.append(cross_val_score(p, X, y, cv=5, scoring="roc_auc").mean())

plt.figure(figsize=(7, 4))
plt.semilogx(C_values, cv_scores, "o-", color="#4C72B0", linewidth=2)
plt.title("Regularisation Strength vs AUC", fontsize=13, fontweight="bold")
plt.xlabel("C (inverse regularisation)", fontsize=12); plt.ylabel("CV ROC-AUC", fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Machine Learning Relevance
# - Used as a **baseline** for any binary classification task
# - Output probabilities are **well-calibrated** (unlike SVM or trees)
# - `C` parameter controls regularisation — important for high-dimensional data
# - Works excellently on linearly separable features after scaling

# %% [markdown]
# ## Common Mistakes
# 1. **Not scaling features** — logistic regression is sensitive to feature scale
# 2. **Ignoring class imbalance** — use `class_weight='balanced'` or SMOTE
# 3. **Using accuracy on imbalanced datasets** — prefer F1, AUC-ROC
# 4. **Ignoring convergence warnings** — increase `max_iter` or scale features
# 5. **Treating probabilities as calibrated without checking** — use calibration plots

# %% [markdown]
# ## Interview Questions
# 1. What is the difference between logistic regression and linear regression?
# 2. Why is log-loss (cross-entropy) used instead of MSE for classification?
# 3. What does the `C` parameter in sklearn's LogisticRegression control?
# 4. How would you handle class imbalance in a logistic regression model?
# 5. What is the difference between one-vs-rest and multinomial logistic regression?

# %% [markdown]
# ## Practice Problems
# 1. Train logistic regression on `load_digits()` and report a full classification report.
# 2. Vary the classification threshold from 0.3 to 0.7 and plot precision vs. recall.
# 3. Compare `solver='lbfgs'` vs `solver='saga'` on the breast cancer dataset.
# 4. Implement logistic regression with L1 penalty and identify the 5 most important features.
# 5. Plot calibration curve (`sklearn.calibration.CalibrationDisplay`) for the cancer classifier.

# %% [markdown]
# ## Solutions

# %%
# Problem 1
from sklearn.datasets import load_digits
digits = load_digits()
Xd, yd = digits.data, digits.target
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
p_digits = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
p_digits.fit(Xd_tr, yd_tr)
print(classification_report(yd_te, p_digits.predict(Xd_te)))

# %%
# Problem 2 — Threshold sweep
from sklearn.metrics import precision_score, recall_score
y_proba = pipe.predict_proba(X_te)[:, 1]
thresholds_sweep = np.arange(0.2, 0.81, 0.05)
precisions, recalls = [], []
for t in thresholds_sweep:
    preds = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_te, preds, zero_division=0))
    recalls.append(recall_score(y_te, preds, zero_division=0))

plt.figure(figsize=(7, 4))
plt.plot(thresholds_sweep, precisions, "o-", label="Precision", color="#4C72B0")
plt.plot(thresholds_sweep, recalls, "s-", label="Recall", color="#DD8452")
plt.title("Precision & Recall vs. Threshold", fontsize=13, fontweight="bold")
plt.xlabel("Threshold", fontsize=12); plt.ylabel("Score", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Further Reading
# - [sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# - [Understanding ROC-AUC](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
# - Géron, *Hands-On ML*, Ch. 4 — Training Models
