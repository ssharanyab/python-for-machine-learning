# %% [markdown]
# # Joining and Concatenating DataFrames
# 
# # Why this matters
# While `merge` combines data horizontally based on column values, `join` does it strictly based on the index. More importantly, `concat` allows you to stack DataFrames vertically (appending rows) or horizontally. Stacking data is vital when you receive data in chunks (e.g., a new CSV file every month) and need to combine them for modeling.
# 
# # Learning Objectives
# 1. Differentiate `.join()` from `.merge()`.
# 2. Master `pd.concat()` to append rows vertically.
# 3. Use `pd.concat()` to attach columns horizontally.
# 
# # Concept Explanation
# - `join`: A convenient method for combining columns of two potentially differently-indexed DataFrames into a single result DataFrame based on their index.
# - `concat`: Stitches DataFrames together along an axis (axis=0 for rows, axis=1 for columns).
# 
# # Beginner Examples

# %%
import pandas as pd

# DataFrames with indices set
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'B': [4, 5, 6]}, index=['a', 'b', 'd'])

# 1. Join (Defaults to Left Join based on Index)
joined_df = df1.join(df2)
print("Default Join (Left):\n", joined_df)

# Outer Join based on Index
outer_joined = df1.join(df2, how='outer')
print("\nOuter Join:\n", outer_joined)

# 2. Concatenation (Vertical - stacking rows)
data_jan = pd.DataFrame({'id': [1, 2], 'sales': [100, 200]})
data_feb = pd.DataFrame({'id': [3, 4], 'sales': [300, 400]})

stacked_data = pd.concat([data_jan, data_feb])
print("\nVertically Concatenated:\n", stacked_data)

# %% [markdown]
# # Intermediate Examples

# %%
# Notice in the stacked data above, the index might repeat (0, 1, 0, 1). 
# Resetting index during concat is cleaner
stacked_clean = pd.concat([data_jan, data_feb], ignore_index=True)
print("Cleanly Concatenated (ignore_index=True):\n", stacked_clean)

# Concatenation (Horizontal - gluing columns)
# This requires the indices to match perfectly, or you will get NaNs.
features_pt1 = pd.DataFrame({'age': [25, 30], 'income': [50k, 60k]}, index=[0, 1])
features_pt2 = pd.DataFrame({'has_car': [True, False]}, index=[0, 1])

# axis=1 means concatenate along columns
full_features = pd.concat([features_pt1, features_pt2], axis=1)
print("\nHorizontally Concatenated:\n", full_features)

# %% [markdown]
# # Machine Learning Relevance
# Concatenation is everywhere in ML.
# 1. **Cross-Validation:** You might split your data into folds, and then `concat` them back together after generating out-of-fold predictions.
# 2. **Pipeline outputs:** Often, numerical features and categorical features are processed in separate pipelines (imputed, scaled, one-hot encoded). You then `concat(axis=1)` to stitch the processed matrices back together before feeding them into the model.
# 
# # Common Mistakes
# - Using `pd.concat(axis=0)` (vertical stack) when the columns don't match exactly. Pandas will create new columns and fill missing values with NaN.
# - Forgetting `ignore_index=True` when stacking rows, leading to duplicate index values which break `.loc` indexing later.
# 
# # Interview Questions
# 1. What is the fundamental difference between `pd.merge()` and `df.join()`?
# 2. How do you append the rows of one DataFrame to another?
# 3. What does the `ignore_index=True` parameter do in `pd.concat()`?
# 4. What happens if you concatenate two DataFrames vertically that have different column names?
# 5. When using `pd.concat(axis=1)`, what dictates how the rows align?
# 
# # Practice Problems
# 1. Create two DataFrames: `q1_sales` and `q2_sales` with the same columns.
# 2. Use `pd.concat` to combine them into `h1_sales`, resetting the index.
# 3. Create a DataFrame `customer_info` indexed by `customer_id`. Create another DataFrame `customer_prefs` also indexed by `customer_id`.
# 4. Use `.join()` to combine the two dataframes based on their index.
# 5. Use `pd.concat` to combine them horizontally. How does this compare to `.join()`?
# 
# # Solutions

# %%
# Problem 1
q1_sales = pd.DataFrame({'month': ['Jan', 'Feb'], 'sales': [100, 150]})
q2_sales = pd.DataFrame({'month': ['Apr', 'May'], 'sales': [200, 250]})

# Problem 2
h1_sales = pd.concat([q1_sales, q2_sales], ignore_index=True)
print("H1 Sales (Concat):\n", h1_sales)

# Problem 3
customer_info = pd.DataFrame({'age': [20, 30]}, index=['C1', 'C2'])
customer_prefs = pd.DataFrame({'color': ['Red', 'Blue']}, index=['C1', 'C2'])

# Problem 4
joined_cust = customer_info.join(customer_prefs)
print("\nJoined Customers:\n", joined_cust)

# Problem 5
concat_cust = pd.concat([customer_info, customer_prefs], axis=1)
print("\nConcat Axis=1 Customers:\n", concat_cust)
# In this case, since indices match perfectly, join and concat(axis=1) produce the same result.

# %% [markdown]
# # Further Reading
# - Pandas Concat documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
