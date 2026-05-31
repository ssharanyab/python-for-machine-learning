# %% [markdown]
# # Sorting DataFrames
# 
# # Why this matters
# Sorting data helps you find the most extreme values (e.g., highest value customers, oldest transactions, features with highest correlation) quickly. It's also often required before visualizing data or performing time-series operations.
# 
# # Learning Objectives
# 1. Master `sort_values()` for sorting by one or multiple columns.
# 2. Master `sort_index()` for sorting by row labels.
# 3. Understand ascending vs descending sorts and dealing with missing values during a sort.
# 
# # Concept Explanation
# Sorting reorganizes the rows of a DataFrame based on the values in one or more columns, or based on the index itself.
# 
# # Beginner Examples

# %%
import pandas as pd
import numpy as np

data = {
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Desk'],
    'price': [1200, 25, 100, 300, 450],
    'rating': [4.8, 4.2, 4.5, 4.0, 4.8]
}
df = pd.DataFrame(data)

# Sort by a single column (ascending by default)
sorted_by_price = df.sort_values(by='price')
print("Sorted by Price (Ascending):\n", sorted_by_price)

# Sort descending
sorted_by_rating_desc = df.sort_values(by='rating', ascending=False)
print("\nSorted by Rating (Descending):\n", sorted_by_rating_desc)

# %% [markdown]
# # Intermediate Examples

# %%
# Sort by multiple columns
# E.g., sort by rating (highest first), then by price (lowest first) for tie-breakers
complex_sort = df.sort_values(by=['rating', 'price'], ascending=[False, True])
print("Complex Sort (Rating Desc, Price Asc):\n", complex_sort)

# Sorting with missing values
df_nan = df.copy()
df_nan.loc[2, 'rating'] = np.nan
print("\nDataFrame with NaN:\n", df_nan)

# Missing values go to the end by default. You can change this with na_position
sort_na_first = df_nan.sort_values(by='rating', na_position='first')
print("\nSorted with NaNs first:\n", sort_na_first)

# Sorting by index
df_indexed = df.set_index('product')
sorted_by_index = df_indexed.sort_index() # Sorts alphabetically by product name
print("\nSorted by Index:\n", sorted_by_index)

# %% [markdown]
# # Machine Learning Relevance
# In ML, ranking metrics (like Feature Importance from a Random Forest) are returned as unsorted arrays or Series. You must sort them to see the top N most important features. Also, for Sequence Models or Recurrent Neural Networks (RNNs) dealing with time-series, ensuring the DataFrame is sorted by time before generating training sequences is absolutely mandatory.
# 
# # Common Mistakes
# - Forgetting that `sort_values` returns a new DataFrame rather than sorting in-place. If you don't reassign it `df = df.sort_values(...)` or use `inplace=True`, your data remains unsorted.
# - Attempting to sort a categorical column alphabetically when it should be sorted logically (e.g., 'Low', 'Medium', 'High'). This requires converting to a Categorical type with an explicit order first.
# 
# # Interview Questions
# 1. How do you sort a DataFrame by two columns, where one is ascending and the other is descending?
# 2. What happens to NaN values by default when you sort a column in descending order?
# 3. What is the difference between `sort_values()` and `sort_index()`?
# 4. Explain how to find the top 3 rows with the highest values in a specific column using sorting.
# 5. Is it generally better to use `inplace=True` or to reassign the DataFrame variable?
# 
# # Practice Problems
# 1. Create a DataFrame with 5 users, containing `age` and `account_balance`.
# 2. Sort the DataFrame by `age` in descending order.
# 3. Get the top 2 users with the highest `account_balance` using `.head()` after sorting.
# 4. Use `.nlargest()` (a specialized sorting method) to find the 2 highest balances directly.
# 5. Sort the DataFrame by its index in descending order.
# 
# # Solutions

# %%
# Problem 1
user_df = pd.DataFrame({
    'age': [22, 45, 30, 45, 60],
    'account_balance': [1500, 3000, 1200, 4000, 8000]
}, index=['U1', 'U2', 'U3', 'U4', 'U5'])

# Problem 2
sort_age = user_df.sort_values('age', ascending=False)
print("Sorted by Age Desc:\n", sort_age)

# Problem 3
top_2_balances = user_df.sort_values('account_balance', ascending=False).head(2)
print("\nTop 2 Balances (sort + head):\n", top_2_balances)

# Problem 4
nlargest_balances = user_df.nlargest(2, 'account_balance')
print("\nTop 2 Balances (nlargest):\n", nlargest_balances)

# Problem 5
sort_idx_desc = user_df.sort_index(ascending=False)
print("\nSorted by Index Desc:\n", sort_idx_desc)

# %% [markdown]
# # Further Reading
# - Pandas Sorting Documentation: https://pandas.pydata.org/docs/user_guide/basics.html#sorting
