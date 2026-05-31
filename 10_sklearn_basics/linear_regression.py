# %% [markdown]
# # Linear Regression
#
# ## Why this matters
# Linear regression is the foundation of supervised ML. It introduces key concepts
# (loss functions, gradient descent, regularisation) that underpin every advanced model.
# It is also an essential interview topic and a go-to baseline model.
#
# ## Learning Objectives
# - Understand the math behind Ordinary Least Squares
# - Train, evaluate, and interpret a linear regression model with sklearn
# - Apply regularisation (Ridge, Lasso)
# - Interpret coefficients and residuals

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## Concept Explanation
# Linear regression models the relationship between a continuous target `y` and features `X`:
#
# `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`
#
# Training minimises the **Mean Squared Error (MSE)** loss:
#
# `MSE = (1/n) Σ (yᵢ - ŷᵢ)²`

# %% [markdown]
# ## Beginner Examples

# %%
# 1. Simple synthetic regression
np.random.seed(42)
X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
y_simple = 3.5 * X_simple.ravel() + 7 + np.random.randn(100) * 2

model = LinearRegression()
model.fit(X_simple, y_simple)
print(f"Coefficient (slope)    : {model.coef_[0]:.3f}  (true: 3.5)")
print(f"Intercept              : {model.intercept_:.3f}  (true: 7.0)")
print(f"R² Score               : {model.score(X_simple, y_simple):.4f}")

# %%
# 2. Visualise fit
y_pred = model.predict(X_simple)
plt.figure(figsize=(8, 5))
plt.scatter(X_simple, y_simple, alpha=0.5, color="#4C72B0", label="Data points")
plt.plot(X_simple, y_pred, color="#DD8452", linewidth=2.5, label=f"Fit: y={model.coef_[0]:.2f}x+{model.intercept_:.2f}")
plt.title("Simple Linear Regression", fontsize=14, fontweight="bold")
plt.xlabel("X", fontsize=12); plt.ylabel("y", fontsize=12)
plt.legend(); plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %%
# 3. California housing — multivariable regression
housing = fetch_california_housing()
Xh, yh = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_hat = lr.predict(X_te)
print(f"Test R²  : {r2_score(y_te, y_hat):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_te, y_hat)):.4f}")

# %%
# 4. Coefficient interpretation
coef_df = pd.DataFrame({
    "Feature": housing.feature_names,
    "Coefficient": lr.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))

# %%
# 5. Residual plot
residuals = y_te - y_hat
plt.figure(figsize=(8, 4))
plt.scatter(y_hat, residuals, alpha=0.4, color="#4C72B0")
plt.axhline(0, color="#DD8452", linewidth=2)
plt.title("Residual Plot — California Housing", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Value", fontsize=12); plt.ylabel("Residual", fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Intermediate Examples

# %%
# 6. Ridge Regression (L2 regularisation)
pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
pipe_ridge.fit(X_tr, y_tr)
print(f"Ridge R²  : {pipe_ridge.score(X_te, y_te):.4f}")

# %%
# 7. Lasso Regression (L1 — sparse coefficients)
pipe_lasso = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.01, max_iter=5000))])
pipe_lasso.fit(X_tr, y_tr)
lasso_coefs = pipe_lasso.named_steps["lasso"].coef_
print("Lasso zeroed features:", np.sum(lasso_coefs == 0))
print(f"Lasso R²  : {pipe_lasso.score(X_te, y_te):.4f}")

# %%
# 8. Compare alpha values for Ridge
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
r2_scores = []
for a in alphas:
    p = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a))])
    p.fit(X_tr, y_tr)
    r2_scores.append(p.score(X_te, y_te))

plt.figure(figsize=(7, 4))
plt.semilogx(alphas, r2_scores, "o-", color="#4C72B0", linewidth=2)
plt.title("Ridge Alpha vs Test R²", fontsize=13, fontweight="bold")
plt.xlabel("Alpha (log scale)", fontsize=12); plt.ylabel("R²", fontsize=12)
plt.grid(alpha=0.4); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Machine Learning Relevance
# - Linear regression is the baseline — always compare complex models against it
# - Ridge/Lasso prevent overfitting on high-dimensional feature spaces
# - Interpreting coefficients helps with feature selection and explainability
# - Residual analysis reveals model misfit (non-linearity, heteroscedasticity)

# %% [markdown]
# ## Common Mistakes
# 1. **Not scaling features** — coefficients become incomparable
# 2. **Using R² alone** — check RMSE and residuals too
# 3. **Ignoring multicollinearity** — inflated coefficients
# 4. **Applying to non-linear relationships** without feature engineering
# 5. **Forgetting to check residual normality** for inference tasks

# %% [markdown]
# ## Interview Questions
# 1. What is the closed-form solution to OLS regression and when does it fail?
# 2. What is the difference between Ridge and Lasso regularisation? When would you choose each?
# 3. How does R² differ from adjusted R²? Can R² be negative?
# 4. What assumptions does linear regression make? How do you test them?
# 5. How would you detect and handle multicollinearity in a regression problem?

# %% [markdown]
# ## Practice Problems
# 1. Generate synthetic data with `y = 2x² + noise` and fit a linear model. Observe underfitting.
# 2. Add polynomial features (`PolynomialFeatures`) and refit. Compare R² scores.
# 3. Fit a Lasso model on California housing; identify which features are zeroed out at different alpha values.
# 4. Implement leave-one-out CV for a small dataset using `cross_val_score`.
# 5. Plot the learning curve (training size vs. MSE) to diagnose bias vs. variance.

# %% [markdown]
# ## Solutions

# %%
# Problem 1 & 2
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(0)
X_p = np.linspace(-3, 3, 80).reshape(-1, 1)
y_p = 2 * X_p.ravel() ** 2 + np.random.randn(80) * 1.5

for deg, color in zip([1, 2, 4], ["#4C72B0", "#DD8452", "#55A868"]):
    pipe = Pipeline([("poly", PolynomialFeatures(deg)), ("lr", LinearRegression())])
    pipe.fit(X_p, y_p)
    print(f"Degree {deg} R²: {pipe.score(X_p, y_p):.4f}")

# %% [markdown]
# ## Further Reading
# - [sklearn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
# - [sklearn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
# - Géron, *Hands-On ML*, Ch. 4 — Training Models
