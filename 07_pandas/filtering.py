# %% [markdown]
# # Filtering DataFrames
# 
# # Why this matters
# You rarely need all the data in a dataset. Filtering allows you to subset data based on specific conditions (e.g., finding all fraudulent transactions, or focusing only on a specific demographic). This is a critical step in isolating cohorts for Machine Learning analysis.
# 
# # Learning Objectives
# 1. Master boolean indexing to filter rows.
# 2. Use multiple conditions (AND, OR, NOT) correctly.
# 3. Understand `.loc` and `.iloc` for robust selection.
# 
# # Concept Explanation
# Filtering relies on passing a boolean mask (a Series of True/False values) to the DataFrame in brackets `df[mask]`. Pandas returns only the rows where the mask is True.
# 
# # Beginner Examples

# %%
import pandas as pd

data = {
    'customer_id': [1, 2, 3, 4, 5],
    'age': [23, 45, 31, 55, 19],
    'spend': [150, 400, 200, 50, 800],
    'is_premium': [False, True, False, False, True]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Simple filtering: Age > 30
over_30 = df[df['age'] > 30]
print("\nCustomers over 30:\n", over_30)

# Filtering based on a boolean column
premium_customers = df[df['is_premium']]
print("\nPremium Customers:\n", premium_customers)

# %% [markdown]
# # Intermediate Examples

# %%
# Multiple conditions: Age > 30 AND spend > 100
# Notice the parentheses around each condition and the bitwise operator &
target_group = df[(df['age'] > 30) & (df['spend'] > 100)]
print("\nOver 30 AND Spend > 100:\n", target_group)

# Using .isin() for matching multiple specific values
specific_customers = df[df['customer_id'].isin([2, 4, 5])]
print("\nSpecific Customers using isin:\n", specific_customers)

# Filtering and selecting specific columns using .loc
# .loc[row_indexer, column_indexer]
young_spends = df.loc[df['age'] < 25, ['customer_id', 'spend']]
print("\nYoung Customers' Spends (using .loc):\n", young_spends)

# Query method (an alternative string-based approach)
high_spend = df.query('spend > 300')
print("\nHigh Spend using query():\n", high_spend)

# %% [markdown]
# # Machine Learning Relevance
# Filtering is essential for splitting data, removing outliers, and creating specialized models. For example, you might build one model for `df[df['country'] == 'US']` and another for Europe. It's also used to filter out rows with severe data corruption before feeding them into a model training pipeline.
# 
# # Common Mistakes
# - Using the Python `and` / `or` operators instead of the bitwise `&` / `|` operators when combining conditions. Pandas requires bitwise operators.
# - Forgetting parentheses around individual conditions when using `&` / `|`, resulting in a `ValueError` due to operator precedence.
# 
# # Interview Questions
# 1. Why do you need parentheses around conditions like `(df['A'] > 0) & (df['B'] < 10)`?
# 2. What is the difference between `.loc` and `.iloc`?
# 3. How would you filter a DataFrame for rows where column 'Name' starts with 'A'?
# 4. Describe how the `.query()` method works in pandas.
# 5. How do you filter a DataFrame to get the rows where the values in column 'X' are not NaN?
# 
# # Practice Problems
# 1. Given the DataFrame `df`, filter for customers who are NOT premium members (using the `~` operator).
# 2. Filter for customers who are either under 25 OR spend over 300.
# 3. Use `.loc` to update the `is_premium` status to `True` for anyone who spent over 200.
# 4. Filter for customers whose `age` is between 30 and 50 inclusive using `.between()`.
# 5. Select the 2nd and 3rd rows of the DataFrame using `.iloc`.
# 
# # Solutions

# %%
# Problem 1
not_premium = df[~df['is_premium']]
print("Not Premium:\n", not_premium)

# Problem 2
young_or_big_spender = df[(df['age'] < 25) | (df['spend'] > 300)]
print("\nYoung or Big Spender:\n", young_or_big_spender)

# Problem 3
# Working on a copy to preserve original for next examples
df_mod = df.copy()
df_mod.loc[df_mod['spend'] > 200, 'is_premium'] = True
print("\nUpdated Premium Status:\n", df_mod)

# Problem 4
mid_age = df[df['age'].between(30, 50)]
print("\nMid-aged Customers:\n", mid_age)

# Problem 5
iloc_selection = df.iloc[1:3] # rows at index 1 and 2
print("\n.iloc Selection:\n", iloc_selection)

# %% [markdown]
# # Further Reading
# - Boolean Indexing in Pandas: https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing
