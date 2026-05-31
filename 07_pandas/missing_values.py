# %% [markdown]
# # Missing Values
# 
# # Why this matters
# Real-world data is messy and almost always contains missing values (NaN/Nulls). Machine Learning algorithms (like Scikit-Learn estimators) generally cannot handle missing values out-of-the-box. Identifying and appropriately handling missing data (imputation or removal) is a mandatory preprocessing step.
# 
# # Learning Objectives
# 1. Identify missing values in a DataFrame using `.isna()` and `.info()`.
# 2. Learn how to drop rows/columns with missing data using `.dropna()`.
# 3. Learn how to impute (fill) missing data using `.fillna()`.
# 
# # Concept Explanation
# Pandas represents missing data primarily with the numpy `NaN` (Not a Number) object. You must decide whether to remove data points with NaNs or fill them in with a calculated value (like the mean, median, or a placeholder constant).
# 
# # Beginner Examples

# %%
import pandas as pd
import numpy as np

data = {
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 2, 3, 4],
    'C': [1, 2, 3, 4]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Identifying missing values
print("\nBoolean mask of missing values:\n", df.isna())
print("\nCount of missing values per column:\n", df.isna().sum())

# Dropping rows with ANY missing values
df_dropped = df.dropna()
print("\nDropped rows with NaNs:\n", df_dropped)

# Filling missing values with a constant (e.g., 0)
df_filled = df.fillna(0)
print("\nFilled NaNs with 0:\n", df_filled)

# %% [markdown]
# # Intermediate Examples

# %%
# Filling with the mean of the column (Simple Imputation)
df_mean_fill = df.copy()
df_mean_fill['A'] = df_mean_fill['A'].fillna(df_mean_fill['A'].mean())
print("Filled Col A with Mean:\n", df_mean_fill)

# Forward fill (propagate the previous valid observation forward)
# Highly useful in time-series data
df_ffill = df.ffill() 
print("\nForward Filled:\n", df_ffill)

# Dropping columns that have too many missing values
# E.g., require at least 3 non-NA values to keep the column
df_thresh = df.dropna(axis=1, thresh=3)
print("\nDropped columns with < 3 non-NA values:\n", df_thresh)

# %% [markdown]
# # Machine Learning Relevance
# Missing value handling (Imputation) is a massive topic in ML. Dropping rows reduces your training data size. Filling with mean/median works for numeric data but reduces variance. Categorical data is often filled with the mode or a new category like "Unknown". More advanced ML techniques use models (like KNNImputer) to predict and fill missing values. Scikit-learn has dedicated classes (e.g., `SimpleImputer`) to integrate this step into ML pipelines, ensuring data leakage does not happen during cross-validation.
# 
# # Common Mistakes
# - Filling missing values *before* doing a train-test split. If you fill NaNs with the mean of the entire dataset, information from the test set "leaks" into the training set. Always compute the mean on the training set and apply it to both.
# - Blindly dropping rows. If 90% of your data has a missing value in a trivial column, dropping rows deletes 90% of your data. Drop the column instead, or impute.
# 
# # Interview Questions
# 1. How does Pandas represent missing numerical data?
# 2. What does `df.isna().sum()` tell you?
# 3. What is the difference between `dropna(axis=0)` and `dropna(axis=1)`?
# 4. How would you fill missing values in a column with the median of that column?
# 5. What is forward filling (`ffill`) and when is it particularly useful?
# 
# # Practice Problems
# 1. Create a DataFrame with columns `id`, `age`, and `income`. Introduce `np.nan` into `age` and `income`.
# 2. Find the total count of missing values in the DataFrame.
# 3. Drop all rows where `income` is missing using the `subset` parameter in `dropna`.
# 4. Fill the missing `age` values with the median age.
# 5. Fill missing values using a backward fill (`bfill`).
# 
# # Solutions

# %%
# Problem 1
df_prac = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'age': [25, np.nan, 35, 40],
    'income': [50000, 60000, np.nan, 80000]
})
print("Original:\n", df_prac)

# Problem 2
print("\nTotal missing values:", df_prac.isna().sum().sum())

# Problem 3
df_drop_inc = df_prac.dropna(subset=['income'])
print("\nDropped missing income:\n", df_drop_inc)

# Problem 4
df_prac['age'] = df_prac['age'].fillna(df_prac['age'].median())
print("\nFilled age with median:\n", df_prac)

# Problem 5
# Reset for demonstration
df_prac.loc[1, 'age'] = np.nan
df_bfill = df_prac.bfill()
print("\nBackward filled:\n", df_bfill)

# %% [markdown]
# # Further Reading
# - Pandas Missing Data: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
