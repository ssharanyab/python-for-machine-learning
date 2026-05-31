# %% [markdown]
# # Variance & Standard Deviation — Measuring Spread
#
# **Module:** 08 — Statistics for Machine Learning
# **Difficulty:** Beginner → Intermediate
# **Prerequisites:** mean_median_mode.py, NumPy, Matplotlib

# %% [markdown]
# ## Why This Matters
#
# Knowing where data is centered is only half the story — you also need to know
# how **spread out** it is. Variance and standard deviation are the canonical
# measures of dispersion and appear everywhere in ML:
#
# - **Standardization (Z-score):** `(x − μ) / σ` — the σ comes from std deviation
# - **Regularization:** L2 penalty penalises large weight *variance*
# - **Bias-Variance Trade-off:** one of the central concepts in all of ML theory
# - **Anomaly detection:** points beyond ±3σ are candidates for outliers
# - **PCA / LDA:** maximising variance in the projected space
# - **Gaussian Naive Bayes:** models each class with μ and σ per feature
#
# Understanding variance at an intuitive level makes the above topics click naturally.

# %% [markdown]
# ## Learning Objectives
#
# 1. Define population variance, sample variance, and standard deviation.
# 2. Explain the n vs. n−1 denominator (Bessel's correction) and when each applies.
# 3. Compute these statistics from scratch and with NumPy/Pandas.
# 4. Apply the empirical rule (68-95-99.7) and the Z-score for outlier detection.
# 5. Use variance and std in real ML preprocessing pipelines.

# %% [markdown]
# ## Concept Explanation
#
# ### Population Variance
# $$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N}(x_i - \mu)^2$$
#
# ### Sample Variance (Bessel's correction)
# $$s^2 = \frac{1}{N-1} \sum_{i=1}^{N}(x_i - \bar{x})^2$$
#
# We divide by N−1 (not N) when estimating the population variance from a **sample**
# to correct for the fact that the sample mean underestimates the true spread.
#
# ### Standard Deviation
# $$\sigma = \sqrt{\sigma^2}$$
#
# Expressed in the **same units** as the original data, making it interpretable.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
print("Libraries loaded ✓")

# %% [markdown]
# ## Beginner Examples

# %% [markdown]
# ### Example 1 — Variance & Std from Scratch

# %%
# ── Example 1: From-scratch implementations ──────────────────────────────────
def population_variance(data):
    mu = sum(data) / len(data)
    return sum((x - mu) ** 2 for x in data) / len(data)

def sample_variance(data):
    mu = sum(data) / len(data)
    return sum((x - mu) ** 2 for x in data) / (len(data) - 1)

def std_dev(variance):
    return variance ** 0.5

heights = [170, 165, 180, 175, 168, 172, 178, 160, 185, 171]

pop_var  = population_variance(heights)
samp_var = sample_variance(heights)

print(f"Data      : {heights}")
print(f"Mean      : {sum(heights)/len(heights):.2f} cm")
print(f"Pop  Var  : {pop_var:.4f}   σ  = {std_dev(pop_var):.4f} cm")
print(f"Samp Var  : {samp_var:.4f}   s  = {std_dev(samp_var):.4f} cm")

# Verify with NumPy (ddof=0 → population, ddof=1 → sample)
print(f"\nNumPy check (pop) : var={np.var(heights, ddof=0):.4f}  std={np.std(heights, ddof=0):.4f}")
print(f"NumPy check (samp): var={np.var(heights, ddof=1):.4f}  std={np.std(heights, ddof=1):.4f}")

# %% [markdown]
# ### Example 2 — Visualising Spread

# %%
# ── Example 2: High vs Low variance distributions ────────────────────────────
low_spread  = np.random.normal(50, 3, 500)
high_spread = np.random.normal(50, 15, 500)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, data, label, color in [
    (axes[0], low_spread,  f"Low σ  ≈ {np.std(low_spread):.1f}",  "#3498db"),
    (axes[1], high_spread, f"High σ ≈ {np.std(high_spread):.1f}", "#e74c3c")
]:
    ax.hist(data, bins=35, color=color, edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(data), color="black", lw=2, linestyle="--", label="Mean")
    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Value")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

axes[0].set_ylabel("Frequency")
plt.suptitle("Effect of Standard Deviation on Distribution Shape", fontsize=14)
plt.tight_layout()
plt.savefig("variance_spread_comparison.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Example 3 — Bessel's Correction Demo

# %%
# ── Example 3: Show why N-1 is unbiased ──────────────────────────────────────
true_population = np.random.normal(100, 15, 100_000)
true_var = np.var(true_population)

sample_sizes = [5, 10, 20, 50, 100]
n_trials = 2000

print(f"True population variance: {true_var:.4f}\n")
print(f"{'n':>5} | {'Avg biased (÷N)':>18} | {'Avg unbiased (÷N-1)':>22}")
print("-" * 50)

for n in sample_sizes:
    biased   = np.mean([np.var(np.random.choice(true_population, n), ddof=0) for _ in range(n_trials)])
    unbiased = np.mean([np.var(np.random.choice(true_population, n), ddof=1) for _ in range(n_trials)])
    print(f"{n:>5} | {biased:>18.4f} | {unbiased:>22.4f}")

print("\n→ Biased estimator consistently underestimates true variance for small samples.")

# %% [markdown]
# ### Example 4 — The Empirical Rule (68-95-99.7)

# %%
# ── Example 4: Empirical rule visualization ───────────────────────────────────
mu, sigma = 100, 15   # IQ scores (approximate)
data_iq   = np.random.normal(mu, sigma, 10_000)

fig, ax = plt.subplots(figsize=(11, 5))

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
from scipy.stats import norm
y = norm.pdf(x, mu, sigma)

# Shade regions
colors   = ["#d5e8f5", "#a8d1f0", "#5ba8e0"]
labels   = ["±1σ  (68.27%)", "±2σ  (95.45%)", "±3σ  (99.73%)"]
sigmas   = [1, 2, 3]

for s, c, lb in zip(reversed(sigmas), reversed(colors), reversed(labels)):
    ax.fill_between(x, y, where=(x >= mu - s*sigma) & (x <= mu + s*sigma),
                    color=c, label=lb, alpha=0.9)

ax.plot(x, y, color="#1a1a2e", lw=2.5)

for s in sigmas:
    ax.axvline(mu + s*sigma, color="gray", lw=1, linestyle=":")
    ax.axvline(mu - s*sigma, color="gray", lw=1, linestyle=":")

ax.set_title("Empirical Rule — The 68-95-99.7 Rule", fontsize=14)
ax.set_xlabel("IQ Score")
ax.set_ylabel("Probability Density")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("empirical_rule.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Intermediate Examples

# %% [markdown]
# ### Example 5 — Z-Score Computation & Outlier Detection

# %%
# ── Example 5: Z-scores ──────────────────────────────────────────────────────
def z_scores(data):
    mu  = np.mean(data)
    sig = np.std(data, ddof=1)
    return (data - mu) / sig

# Transaction amounts — a few fraudulent extremes
transactions = np.concatenate([
    np.random.normal(200, 40, 200),
    np.array([1500, 2200, -800])    # suspicious outliers
])

z = z_scores(transactions)
outliers = transactions[np.abs(z) > 3]

print(f"Total transactions  : {len(transactions)}")
print(f"Outliers (|z| > 3)  : {len(outliers)}")
print(f"Outlier values      : {outliers}")
print(f"\nZ-score range: [{z.min():.2f}, {z.max():.2f}]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.hist(transactions, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
ax1.set_title("Raw Transactions")
ax1.set_xlabel("Amount (USD)")

colors_z = ["#e74c3c" if abs(zi) > 3 else "#2ecc71" for zi in z]
ax2.scatter(range(len(z)), z, c=colors_z, alpha=0.6, s=15)
ax2.axhline( 3, color="#e74c3c", lw=1.5, linestyle="--", label="+3σ threshold")
ax2.axhline(-3, color="#e74c3c", lw=1.5, linestyle="--", label="-3σ threshold")
ax2.axhline( 0, color="black", lw=1)
ax2.set_title("Z-Scores — Outliers in Red")
ax2.set_xlabel("Index")
ax2.set_ylabel("Z-Score")
ax2.legend()

plt.tight_layout()
plt.savefig("zscore_outlier_detection.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Example 6 — Coefficient of Variation (CV)

# %%
# ── Example 6: Coefficient of Variation ──────────────────────────────────────
# CV = σ/μ — unit-free measure; lets you compare spread of different scales
stocks = {
    "Large-Cap ETF":    np.random.normal(100, 5,  252),   # low volatility
    "Tech Growth":      np.random.normal( 80, 18, 252),   # high volatility
    "Cryptocurrency":   np.random.normal( 30, 25, 252),   # extreme volatility
}

print(f"{'Asset':>20} | {'Mean':>8} | {'Std':>8} | {'CV (%)':>8}")
print("-" * 55)
for name, data in stocks.items():
    mu  = np.mean(data)
    sig = np.std(data)
    cv  = (sig / mu) * 100
    print(f"{name:>20} | {mu:>8.2f} | {sig:>8.2f} | {cv:>7.2f}%")

print("\nCV normalises std by mean → fair comparison across different scales.")

# %% [markdown]
# ### Example 7 — Running / Online Variance (Welford's Algorithm)

# %%
# ── Example 7: Welford's online algorithm ────────────────────────────────────
# Useful for streaming data — computes mean/variance in one pass, numerically stable
class OnlineStats:
    """Welford's one-pass online mean and variance calculator."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta  = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self):
        return self.variance ** 0.5

stream = np.random.normal(50, 10, 1000)
tracker = OnlineStats()
for val in stream:
    tracker.update(val)

print(f"Online  → mean={tracker.mean:.4f}  std={tracker.std:.4f}")
print(f"NumPy   → mean={np.mean(stream):.4f}  std={np.std(stream, ddof=1):.4f}")
print("Results match ✓ — useful for streaming / out-of-core ML pipelines.")

# %% [markdown]
# ## Machine Learning Relevance

# %% [markdown]
# ### ML Application 1 — StandardScaler (Z-score Normalization)

# %%
# ── ML App 1: StandardScaler from scratch ────────────────────────────────────
class StandardScaler:
    """Replicates sklearn's StandardScaler behaviour."""
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0, ddof=0)   # population std (sklearn default)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Simulate multi-feature dataset
np.random.seed(7)
X = np.column_stack([
    np.random.normal(5_000, 1_200, 300),   # income (large scale)
    np.random.normal(   35,     8, 300),   # age
    np.random.uniform( 1.5,   5.0, 300),   # rating
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Before Scaling ===")
print(f"Mean : {np.mean(X, axis=0)}")
print(f"Std  : {np.std(X, axis=0)}")

print("\n=== After Scaling (should be ≈0 mean, 1 std) ===")
print(f"Mean : {np.mean(X_scaled, axis=0).round(6)}")
print(f"Std  : {np.std(X_scaled, axis=0).round(6)}")

# %% [markdown]
# ### ML Application 2 — Bias-Variance Trade-off Illustration

# %%
# ── ML App 2: Bias-variance illustrated with polynomial regression ────────────
from numpy.polynomial import polynomial as P

np.random.seed(0)
x_true = np.linspace(0, 1, 200)
y_true = np.sin(2 * np.pi * x_true)

def train_predict(degree, x_train, y_train, x_test):
    coeffs = np.polyfit(x_train, y_train, degree)
    return np.polyval(coeffs, x_test)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
degrees = [1, 5, 15]

for ax, deg in zip(axes, degrees):
    predictions = []
    for _ in range(50):
        idx = np.random.choice(len(x_true), 20, replace=False)
        x_s, y_s = x_true[idx], y_true[idx] + np.random.normal(0, 0.2, 20)
        pred = train_predict(deg, x_s, y_s, x_true)
        predictions.append(pred)
        ax.plot(x_true, pred, color="#3498db", alpha=0.12, lw=1)

    mean_pred = np.mean(predictions, axis=0)
    var_pred  = np.var(predictions, axis=0)

    ax.plot(x_true, y_true, "k-", lw=2, label="True")
    ax.plot(x_true, mean_pred, "r--", lw=2, label="Mean pred")
    ax.set_title(f"Degree {deg}\nVar={np.mean(var_pred):.4f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(-3, 3)

plt.suptitle("Bias-Variance Trade-off — Variance of Predictions", fontsize=13)
plt.tight_layout()
plt.savefig("bias_variance_tradeoff.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### ML Application 3 — Feature Variance for Feature Selection

# %%
# ── ML App 3: Variance threshold feature selection ───────────────────────────
from sklearn.feature_selection import VarianceThreshold

np.random.seed(1)
X_fs = np.column_stack([
    np.random.normal(0, 1, 200),          # useful feature (high var)
    np.random.normal(0, 0.01, 200),       # near-constant (low var) — useless
    np.random.choice([0, 1], 200),        # binary — moderate var
    np.ones(200),                         # constant — var = 0, useless
])

selector = VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X_fs)

print(f"Original features : {X_fs.shape[1]}")
print(f"After threshold   : {X_filtered.shape[1]}")
print(f"Variances         : {np.var(X_fs, axis=0).round(4)}")
print(f"Selected mask     : {selector.get_support()}")
print("\n→ Low-variance features contribute little signal and increase overfitting risk.")

# %% [markdown]
# ## Common Mistakes

# %%
# ── Common Mistakes ───────────────────────────────────────────────────────────

# Mistake 1: ddof confusion
data_m = [2, 4, 4, 4, 5, 5, 7, 9]
print("=== Mistake 1: Mixing up ddof ===")
print(f"np.var(ddof=0) — population: {np.var(data_m, ddof=0):.4f}")
print(f"np.var(ddof=1) — sample    : {np.var(data_m, ddof=1):.4f}")
print("sklearn scalers use ddof=0 (population); pandas .std() defaults to ddof=1 (sample)\n")

# Mistake 2: Std in different units vs variance
std_cm = np.std([170, 175, 168, 180])
print(f"=== Mistake 2: Variance units ===")
print(f"Std dev of heights: {std_cm:.2f} cm  ← interpretable")
print(f"Variance of heights: {std_cm**2:.2f} cm²  ← hard to interpret directly\n")

# Mistake 3: Using std on ordinal data
print("=== Mistake 3: Std on Likert scale ratings ===")
ratings = pd.Series([1, 1, 2, 3, 4, 5, 5])
print(f"Std of Likert ratings: {ratings.std():.2f}")
print("→ Std is valid here, but always check if data is truly ordinal-continuous.\n")

# Mistake 4: Not fitting scaler on train only
print("=== Mistake 4: Data leakage in scaling ===")
print("WRONG: scaler.fit(X_all)  → leaks test statistics into training")
print("RIGHT: scaler.fit(X_train) → scaler.transform(X_train), scaler.transform(X_test)")

# %% [markdown]
# ## Interview Questions
#
# **Q1.** What is Bessel's correction and why is it needed?
#
# > **Answer:** Dividing by N−1 instead of N corrects for the bias introduced when
# > estimating population variance from a sample. The sample mean is derived from the
# > same data, which causes the squared deviations to systematically underestimate
# > the true variance. N−1 makes the estimator unbiased.
#
# **Q2.** sklearn's `StandardScaler` divides by `std(ddof=0)` while Pandas defaults
# to `std(ddof=1)`. Why does this matter?
#
# > **Answer:** For large samples the difference is negligible. For small datasets it
# > can introduce a small scaling inconsistency. Always ensure consistency between
# > your manual preprocessing and library defaults.
#
# **Q3.** What does a Z-score of +3.5 tell you about a data point?
#
# > **Answer:** The point is 3.5 standard deviations above the mean. Under a Normal
# > distribution, fewer than 0.023% of points are that far out — making it a strong
# > outlier candidate.
#
# **Q4.** Explain the bias-variance trade-off using the formulas for each term.
#
# > **Answer:** Expected MSE = Bias² + Variance + Irreducible Noise. Bias is the
# > error from wrong assumptions; Variance is sensitivity to training-set fluctuations.
# > Complex models have low bias / high variance; simple models the reverse.
#
# **Q5.** When should you use `VarianceThreshold` for feature selection?
#
# > **Answer:** As a quick preprocessing step to remove near-constant features
# > (features that carry almost no information). Set the threshold based on domain
# > knowledge; a common default is 0 (removes only perfectly constant features).

# %% [markdown]
# ## Practice Problems
#
# **Problem 1.** Compute population and sample variance of `[2, 4, 4, 4, 5, 5, 7, 9]`
# by hand; verify with NumPy.
#
# **Problem 2.** Generate two Normal distributions with the same mean (μ=50) but
# different σ (5 and 20). Plot overlapping histograms and annotate with empirical-rule
# bounds for each.
#
# **Problem 3.** Implement a `RobustScaler` that uses median and IQR instead of
# mean and std. Apply to a dataset with outliers; compare to `StandardScaler`.
#
# **Problem 4.** Implement Welford's online algorithm from scratch and verify it
# gives the same variance as NumPy on a stream of 5 000 random values.
#
# **Problem 5.** Given features with variances `[0.001, 1.5, 0.0, 3.2, 0.05]`,
# determine which features to keep using `VarianceThreshold(threshold=0.1)`.

# %% [markdown]
# ## Solutions

# %%
# ── Solution 1 ────────────────────────────────────────────────────────────────
d = [2, 4, 4, 4, 5, 5, 7, 9]
mu_d = sum(d) / len(d)
pop_v  = sum((x - mu_d)**2 for x in d) / len(d)
samp_v = sum((x - mu_d)**2 for x in d) / (len(d) - 1)
print(f"Pop  var: {pop_v:.4f}  ← np.var ddof=0: {np.var(d, ddof=0):.4f}")
print(f"Samp var: {samp_v:.4f}  ← np.var ddof=1: {np.var(d, ddof=1):.4f}")

# ── Solution 3: RobustScaler ──────────────────────────────────────────────────
class RobustScaler:
    def fit(self, X):
        self.median_ = np.median(X, axis=0)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        self.iqr_ = Q3 - Q1
        return self
    def transform(self, X):
        return (X - self.median_) / self.iqr_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

Xr = np.concatenate([np.random.normal(50, 10, 100), [200, 250, -100]]).reshape(-1, 1)
rs = RobustScaler().fit_transform(Xr)
ss = StandardScaler().fit_transform(Xr)
print(f"\nRobust → std of scaled data: {np.std(rs):.4f}")
print(f"Standard→ std of scaled data: {np.std(ss):.4f}")
print("RobustScaler is less influenced by the outliers.")

# ── Solution 5 ────────────────────────────────────────────────────────────────
variances = np.array([0.001, 1.5, 0.0, 3.2, 0.05])
threshold = 0.1
kept = variances >= threshold
print(f"\nVariances: {variances}")
print(f"Kept mask: {kept}")
print(f"Features kept (indices): {np.where(kept)[0]}")

# %% [markdown]
# ## Further Reading
#
# - **Book:** "The Elements of Statistical Learning" — Hastie, Tibshirani, Friedman (Ch. 2)
# - **sklearn:** [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# - **sklearn:** [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
# - **sklearn:** [`VarianceThreshold`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
# - **Article:** "Welford's Online Algorithm" — Wikipedia
# - **Video:** StatQuest — "Bias and Variance" by Josh Starmer
