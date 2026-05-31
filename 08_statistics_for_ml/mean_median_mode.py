# %% [markdown]
# # Mean, Median & Mode — Measures of Central Tendency
#
# **Module:** 08 — Statistics for Machine Learning
# **Difficulty:** Beginner → Intermediate
# **Prerequisites:** Python basics, NumPy, Matplotlib

# %% [markdown]
# ## Why This Matters
#
# Every ML pipeline begins with **understanding your data**. Mean, median, and mode
# are the first three statistics you compute — they tell you where the "center" of
# your data lives. Without them you cannot:
# - Detect skewed feature distributions before training
# - Impute missing values intelligently
# - Understand label imbalance in classification tasks
# - Scale features properly for gradient-based optimizers
#
# These three numbers are deceptively simple yet routinely misused in industry,
# leading to subtle model degradation that is hard to trace back to its root cause.

# %% [markdown]
# ## Learning Objectives
#
# By the end of this lesson you will be able to:
# 1. Define and compute mean, median, and mode from scratch and with libraries.
# 2. Explain when each measure is the most appropriate summary statistic.
# 3. Identify skewed distributions and choose robust imputation strategies.
# 4. Apply central tendency concepts to real ML workflows (feature engineering, EDA).
# 5. Recognize common pitfalls such as the effect of outliers on the mean.

# %% [markdown]
# ## Concept Explanation
#
# ### Arithmetic Mean
# The **mean** (μ) is the sum of all values divided by the count:
#
# $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
#
# Sensitive to outliers — a single extreme value can pull it far from the bulk.
#
# ### Median
# The **median** is the middle value when data is sorted. It splits the distribution
# into two equal halves (50th percentile). Robust to outliers.
#
# ### Mode
# The **mode** is the most frequently occurring value. Useful for categorical data
# and multi-modal distributions.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)

print("Libraries loaded ✓")
print(f"NumPy  : {np.__version__}")
print(f"Pandas : {pd.__version__}")

# %% [markdown]
# ## Beginner Examples

# %% [markdown]
# ### Example 1 — Computing the Three Measures from Scratch

# %%
# ── Example 1: From-scratch implementations ──────────────────────────────────
def mean(data):
    """Arithmetic mean."""
    return sum(data) / len(data)

def median(data):
    """Median — middle value of sorted data."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return sorted_data[mid]

def mode(data):
    """Mode — most frequent value(s)."""
    from collections import Counter
    counts = Counter(data)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]

# Simple dataset
scores = [85, 92, 78, 95, 88, 92, 76, 88, 92, 99]
print(f"Scores  : {scores}")
print(f"Mean    : {mean(scores):.2f}")
print(f"Median  : {median(scores)}")
print(f"Mode    : {mode(scores)}")

# %% [markdown]
# ### Example 2 — Using NumPy and SciPy

# %%
# ── Example 2: Library functions ──────────────────────────────────────────────
data = np.array(scores)
print(f"np.mean   : {np.mean(data):.2f}")
print(f"np.median : {np.median(data):.2f}")
print(f"scipy mode: {stats.mode(data, keepdims=True).mode[0]}")

# Pandas equivalent
s = pd.Series(data)
print(f"\nPandas describe:\n{s.describe()}")

# %% [markdown]
# ### Example 3 — Effect of Outliers

# %%
# ── Example 3: Outlier sensitivity ───────────────────────────────────────────
normal  = [50_000, 52_000, 48_000, 55_000, 51_000]
outlier = normal + [5_000_000]   # one CEO salary

print("=== Annual Salaries (USD) ===")
print(f"Without outlier → mean: ${np.mean(normal):>10,.0f}  median: ${np.median(normal):>10,.0f}")
print(f"With outlier    → mean: ${np.mean(outlier):>10,.0f}  median: ${np.median(outlier):>10,.0f}")
print("\nThe median barely changes; the mean skyrockets. ⚠️")

# %% [markdown]
# ### Example 4 — Visualising Central Tendency on a Histogram

# %%
# ── Example 4: Histogram + annotations ───────────────────────────────────────
incomes = np.concatenate([
    np.random.normal(55_000, 8_000, 500),
    np.array([350_000, 400_000, 500_000])   # high earners
])

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(incomes, bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)

mu  = np.mean(incomes)
med = np.median(incomes)
mo  = stats.mode(incomes, keepdims=True).mode[0]

for val, label, color in [(mu, f"Mean\n${mu:,.0f}", "#e74c3c"),
                           (med, f"Median\n${med:,.0f}", "#2ecc71")]:
    ax.axvline(val, color=color, lw=2.5, linestyle="--")
    ax.text(val + 1500, ax.get_ylim()[1] * 0.85, label,
            color=color, fontsize=10, fontweight="bold")

ax.set_title("Income Distribution — Mean vs Median", fontsize=14, pad=12)
ax.set_xlabel("Annual Income (USD)")
ax.set_ylabel("Frequency")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("mean_median_skew.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved → mean_median_skew.png")

# %% [markdown]
# ## Intermediate Examples

# %% [markdown]
# ### Example 5 — Weighted Mean (Feature Importance Analogy)

# %%
# ── Example 5: Weighted mean ──────────────────────────────────────────────────
# Think of weights as feature importance scores
features = np.array([0.85, 0.72, 0.91, 0.60, 0.78])  # model scores
weights  = np.array([0.30, 0.20, 0.25, 0.10, 0.15])   # feature importances

weighted_mean = np.average(features, weights=weights)
simple_mean   = np.mean(features)

print(f"Simple   mean of scores: {simple_mean:.4f}")
print(f"Weighted mean of scores: {weighted_mean:.4f}")
print("\nWeighted mean rewards high-importance features — used in ensemble voting.")

# %% [markdown]
# ### Example 6 — Trimmed Mean (Robust Statistic)

# %%
# ── Example 6: Trimmed mean ───────────────────────────────────────────────────
# Trim top/bottom 5% to eliminate extreme outliers
noisy_data = np.concatenate([np.random.normal(100, 15, 1000),
                              np.random.uniform(500, 1000, 30)])

regular = np.mean(noisy_data)
trimmed = stats.trim_mean(noisy_data, proportiontocut=0.05)

print(f"Regular mean  : {regular:.2f}")
print(f"Trimmed mean  : {trimmed:.2f}")
print(f"True mean ~100: 100.00")
print("\nTrimmed mean is closer to the true mean — used in robust scalers.")

# %% [markdown]
# ### Example 7 — Mode for Categorical Data

# %%
# ── Example 7: Mode in categorical context ───────────────────────────────────
categories = pd.Series(["cat", "dog", "cat", "bird", "cat", "dog", "cat"])
print(f"Mode (most common pet): {categories.mode().values}")
print(f"Value counts:\n{categories.value_counts()}")

# Multi-modal example
multi = pd.Series([1, 2, 2, 3, 3, 4])
print(f"\nMulti-modal data modes: {multi.mode().values}")

# %% [markdown]
# ## Machine Learning Relevance

# %% [markdown]
# ### ML Application 1 — Missing Value Imputation Strategy

# %%
# ── ML App 1: Smart imputation ────────────────────────────────────────────────
np.random.seed(0)
# Simulate a housing dataset with missing values
df = pd.DataFrame({
    "price":    np.random.normal(300_000, 60_000, 200),
    "bedrooms": np.random.choice([2, 3, 3, 3, 4, 5], 200),
    "location": np.random.choice(["A", "B", "B", "C"], 200)
})

# Introduce missingness
df.loc[np.random.choice(df.index, 20, replace=False), "price"]    = np.nan
df.loc[np.random.choice(df.index, 15, replace=False), "bedrooms"] = np.nan
df.loc[np.random.choice(df.index, 10, replace=False), "location"] = np.nan

print("Missing values before imputation:")
print(df.isnull().sum())

# Strategy:
# - Numeric + symmetric  → mean imputation
# - Numeric + skewed     → median imputation
# - Categorical          → mode imputation
df["price_imputed"]    = df["price"].fillna(df["price"].median())    # skewed → median
df["bedrooms_imputed"] = df["bedrooms"].fillna(df["bedrooms"].mean()) # roughly normal → mean
df["location_imputed"] = df["location"].fillna(df["location"].mode()[0])  # categorical → mode

print("\nMissing values after imputation:")
print(df[["price_imputed", "bedrooms_imputed", "location_imputed"]].isnull().sum())

# %% [markdown]
# ### ML Application 2 — Feature Distribution Analysis Before Scaling

# %%
# ── ML App 2: Skewness check before choosing scaler ──────────────────────────
features_df = pd.DataFrame({
    "age":     np.random.normal(35, 8, 500),
    "income":  np.random.exponential(50_000, 500),
    "tenure":  np.random.uniform(0, 30, 500)
})

print("Distribution summary:")
print(features_df.describe().round(2))
print("\nSkewness:")
for col in features_df.columns:
    sk = features_df[col].skew()
    gap = abs(features_df[col].mean() - features_df[col].median())
    recommendation = "StandardScaler" if abs(sk) < 0.5 else "RobustScaler / log-transform"
    print(f"  {col:8s}: skew={sk:+.3f}  mean-median gap={gap:,.1f} → {recommendation}")

# %% [markdown]
# ### ML Application 3 — Label Distribution (Classification)

# %%
# ── ML App 3: Class imbalance detection ──────────────────────────────────────
labels = np.random.choice([0, 1], p=[0.85, 0.15], size=1000)
s = pd.Series(labels)

print("Label distribution:")
print(s.value_counts())
print(f"\nMode (majority class): {s.mode()[0]}")
print(f"Class imbalance ratio: {s.value_counts()[0] / s.value_counts()[1]:.1f}:1")
print("\n→ Imbalanced dataset: consider SMOTE, class_weight='balanced', or oversampling.")

# %% [markdown]
# ## Common Mistakes

# %%
# ── Common Mistakes ───────────────────────────────────────────────────────────

# Mistake 1: Using mean on skewed data for imputation
print("=== Mistake 1: Mean imputation on skewed data ===")
house_prices = np.concatenate([np.random.normal(300_000, 50_000, 100),
                                np.array([2_000_000, 3_500_000])])
print(f"Mean  : ${np.mean(house_prices):>12,.0f}  ← pulled high by mansions")
print(f"Median: ${np.median(house_prices):>12,.0f}  ← robust center\n")

# Mistake 2: Ignoring multi-modality
print("=== Mistake 2: Single mode on bimodal data ===")
bimodal = np.concatenate([np.random.normal(30, 3, 200),
                           np.random.normal(70, 3, 200)])
print(f"Mean  : {np.mean(bimodal):.1f}  ← in the valley between peaks!")
print(f"Median: {np.median(bimodal):.1f}  ← also in the valley!")
print("→ Always plot your data; a single summary stat can be very misleading.\n")

# Mistake 3: Averaging proportions incorrectly (Simpson's Paradox)
print("=== Mistake 3: Naively averaging group means ===")
g1 = pd.DataFrame({"score": [80, 85, 90], "group": "A"})
g2 = pd.DataFrame({"score": [60, 65, 70, 75, 80], "group": "B"})
combined = pd.concat([g1, g2])
print(f"Group A mean : {g1['score'].mean():.1f}")
print(f"Group B mean : {g2['score'].mean():.1f}")
print(f"Grand mean   : {combined['score'].mean():.1f}  (not the average of 85 and 70!)")

# %% [markdown]
# ## Interview Questions
#
# **Q1.** When would you prefer the median over the mean as an imputation strategy, and why?
#
# > **Answer:** Use the median when the feature is skewed or contains outliers. The
# > median is the 50th percentile and is unaffected by extreme values, while the mean
# > is pulled toward them. Example: `income` in a dataset that includes billionaires.
#
# **Q2.** What does it mean for a distribution to be "positively skewed," and how does
# it relate to mean vs. median?
#
# > **Answer:** Positively (right) skewed means a long tail extends to the right.
# > In this case *mean > median > mode*. The mean is pulled toward the tail.
#
# **Q3.** How does the mode relate to class imbalance in classification problems?
#
# > **Answer:** The majority class is the mode of the label distribution. A naive
# > classifier that always predicts the mode achieves the "majority class baseline"
# > accuracy — a useful sanity-check benchmark.
#
# **Q4.** What is the weighted mean, and where does it appear in ML?
#
# > **Answer:** Weighted mean assigns importance weights to each observation.
# > Appears in: ensemble soft-voting, class-weighted loss functions, and
# > sample-weight arguments in sklearn estimators.
#
# **Q5.** Can the mean, median, and mode all be equal? If so, when?
#
# > **Answer:** Yes — in a perfectly symmetric unimodal distribution such as the
# > Normal (Gaussian) distribution. This is actually a key assumption checked
# > before applying many parametric statistical tests.

# %% [markdown]
# ## Practice Problems

# %% [markdown]
# **Problem 1.** Given the array `[4, 8, 6, 5, 3, 2, 8, 9, 2, 5]`,
# compute the mean, median, and mode without using any library.
#
# **Problem 2.** A dataset of house prices (USD) is:
# `[150k, 160k, 155k, 170k, 165k, 2_000k]`. Compare mean vs. median imputation
# for a missing value and explain which is better.
#
# **Problem 3.** Generate 1 000 samples from an exponential distribution
# (λ=1/50000). Compute mean and median. Plot a histogram and annotate both.
#
# **Problem 4.** You have a DataFrame with a column `age` that is 10% missing.
# Skewness of `age` is 0.12. Write code to choose and apply the right imputation.
#
# **Problem 5.** In a binary classification task, 950 samples are class 0 and
# 50 are class 1. What is the mode label? What accuracy does a mode-baseline
# classifier achieve?

# %% [markdown]
# ## Solutions

# %%
# ── Solution 1 ────────────────────────────────────────────────────────────────
from collections import Counter

data_p1 = [4, 8, 6, 5, 3, 2, 8, 9, 2, 5]
mean_p1   = sum(data_p1) / len(data_p1)
sorted_p1 = sorted(data_p1)
n = len(sorted_p1)
median_p1 = (sorted_p1[n//2 - 1] + sorted_p1[n//2]) / 2 if n % 2 == 0 else sorted_p1[n//2]
count_p1  = Counter(data_p1)
max_c     = max(count_p1.values())
mode_p1   = [k for k, v in count_p1.items() if v == max_c]

print(f"Mean: {mean_p1}, Median: {median_p1}, Mode: {mode_p1}")

# ── Solution 2 ────────────────────────────────────────────────────────────────
prices = np.array([150_000, 160_000, 155_000, 170_000, 165_000, 2_000_000])
print(f"\nMean  imputation: ${np.mean(prices):>12,.0f}  ← skewed by mansion")
print(f"Median imputation: ${np.median(prices):>12,.0f}  ← representative center")
print("→ Median is better for this skewed distribution.")

# ── Solution 3 ────────────────────────────────────────────────────────────────
exp_data = np.random.exponential(50_000, 1000)
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(exp_data, bins=40, color="#9b59b6", edgecolor="white", alpha=0.8)
ax.axvline(np.mean(exp_data),   color="#e74c3c", lw=2, label=f"Mean   {np.mean(exp_data):,.0f}")
ax.axvline(np.median(exp_data), color="#2ecc71", lw=2, linestyle="--", label=f"Median {np.median(exp_data):,.0f}")
ax.set_title("Exponential Distribution — Mean vs Median")
ax.legend()
plt.tight_layout()
plt.savefig("solution3_exp_dist.png", dpi=110, bbox_inches="tight")
plt.show()

# ── Solution 4 ────────────────────────────────────────────────────────────────
df_age = pd.DataFrame({"age": np.random.normal(35, 8, 200)})
df_age.loc[np.random.choice(df_age.index, 20), "age"] = np.nan
skew = df_age["age"].skew()
print(f"\nAge skewness: {skew:.3f}")
impute_val = df_age["age"].mean() if abs(skew) < 0.5 else df_age["age"].median()
strategy   = "mean" if abs(skew) < 0.5 else "median"
df_age["age"] = df_age["age"].fillna(impute_val)
print(f"Used {strategy} imputation: {impute_val:.2f}")

# ── Solution 5 ────────────────────────────────────────────────────────────────
labels_p5 = np.array([0]*950 + [1]*50)
mode_label = stats.mode(labels_p5, keepdims=True).mode[0]
baseline_acc = (labels_p5 == mode_label).mean()
print(f"\nMode label: {mode_label}")
print(f"Baseline accuracy (always predict mode): {baseline_acc:.1%}")
print("→ 95% accuracy — sounds great, but precision/recall for class 1 is 0!")

# %% [markdown]
# ## Further Reading
#
# - **Book:** "Statistics" by Freedman, Pisani & Purves — Chapter 4
# - **Book:** "Hands-On ML with Scikit-Learn" by Aurélien Géron — Chapter 2 (EDA)
# - **sklearn:** [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
# - **Article:** "Mean, Median, Mode — When to Use Which" — Towards Data Science
# - **Wikipedia:** [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
# - **Video:** StatQuest — "Mean vs Median" by Josh Starmer
