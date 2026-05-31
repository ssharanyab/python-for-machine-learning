# %% [markdown]
# # NumPy Statistics
# 
# # Why this matters
# Statistical properties of your data form the foundation of Exploratory Data Analysis (EDA) and Machine Learning. Understanding distributions, variances, and correlations helps in selecting features and diagnosing model behavior.
# 
# # Learning Objectives
# - Calculate mean, median, standard deviation, and variance.
# - Understand percentiles and quantiles.
# - Compute correlation coefficients and covariance matrices.
# 
# # Concept Explanation
# - `np.median`: Finds the middle value. Robust to outliers compared to the mean.
# - `np.var`, `np.std`: Measure the spread of the data.
# - `np.percentile`: Finds the value below which a given percentage of observations fall.
# - `np.corrcoef`: Computes the Pearson correlation coefficient matrix.
# 
# # Beginner Examples
# %%
import numpy as np

data = np.array([1, 2, 3, 4, 100]) # 100 is an outlier

# Example 1: Mean vs Median
print(f"Mean: {np.mean(data)}, Median: {np.median(data)}")

# Example 2: Variance and Standard Deviation
print(f"Variance: {np.var(data):.2f}, Std Dev: {np.std(data):.2f}")

# Example 3: Percentiles
print(f"25th percentile: {np.percentile(data, 25)}")
print(f"75th percentile: {np.percentile(data, 75)}")

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Tabular Data - Outlier Detection (IQR Method)
X = np.random.normal(0, 1, 1000)
X = np.append(X, [10, -10]) # Add outliers
Q1 = np.percentile(X, 25)
Q3 = np.percentile(X, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = X[(X < lower_bound) | (X > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")

# Example 5: Feature Correlation
# Two features that are highly correlated
feature_A = np.linspace(0, 10, 100)
feature_B = feature_A * 2 + np.random.normal(0, 1, 100)
correlation_matrix = np.corrcoef(feature_A, feature_B)
print("\nCorrelation Matrix:\n", correlation_matrix)

# Example 6: Image Histogram Stats
image = np.random.randint(0, 256, (128, 128))
print(f"\nImage Pixel Median: {np.median(image)}")
print(f"Image Pixel Std: {np.std(image):.2f}")

# %% [markdown]
# # Machine Learning Relevance
# Normalizing data (using mean and std) is a standard preprocessing step. Checking feature correlation helps in eliminating redundant features to prevent multicollinearity in linear models.
# 
# # Common Mistakes
# - Confusing sample standard deviation vs population standard deviation (`ddof` parameter in NumPy is 0 by default, Pandas is 1).
# - Passing a 2D matrix to `np.corrcoef` without understanding if rows or columns represent variables (variables are rows by default).
# 
# # Interview Questions
# 1. Why might you use median instead of mean to impute missing values?
# 2. What is the difference between `np.std(x)` in NumPy and `x.std()` in Pandas?
# 3. How do you calculate the interquartile range (IQR) in NumPy?
# 4. What does a correlation coefficient of -1 indicate?
# 5. How would you compute the covariance between two 1D arrays?
# 
# # Practice Problems
# 1. Calculate the mean, median, and std of a random array of size 100.
# 2. Find the 90th percentile of an array of 50 values.
# 3. Create two 1D arrays of size 20. Calculate their covariance matrix using `np.cov`.
# 4. Given a feature matrix `(200, 3)`, compute the correlation matrix. (Hint: use `rowvar=False`).
# 5. Impute the `np.nan` values in an array with the median of the non-nan values (Hint: use `np.nanmedian`).
# 
# # Solutions
# %%
# Solution 1
arr = np.random.rand(100)
print("Mean:", np.mean(arr), "Median:", np.median(arr), "Std:", np.std(arr))

# Solution 2
arr2 = np.random.rand(50)
print("90th percentile:", np.percentile(arr2, 90))

# Solution 3
a1 = np.random.rand(20)
a2 = a1 + np.random.rand(20)
print("Covariance:\n", np.cov(a1, a2))

# Solution 4
X = np.random.rand(200, 3)
print("Correlation:\n", np.corrcoef(X, rowvar=False))

# Solution 5
arr_nan = np.array([1, 2, np.nan, 4, 5])
med = np.nanmedian(arr_nan)
arr_nan[np.isnan(arr_nan)] = med
print("Imputed array:", arr_nan)

# %% [markdown]
# # Further Reading
# - NumPy Documentation: `numpy.percentile`, `numpy.corrcoef`, `numpy.nanmedian`.
