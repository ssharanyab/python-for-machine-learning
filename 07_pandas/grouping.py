# %% [markdown]
# # Grouping Data (groupby)
# 
# # Why this matters
# Grouping data follows the "split-apply-combine" paradigm. It allows you to split your data into distinct categories, calculate metrics (like mean, sum, or count) for each category, and combine the results into a new summary DataFrame. This is arguably the most powerful analysis tool in pandas.
# 
# # Learning Objectives
# 1. Understand the `.groupby()` mechanism.
# 2. Learn how to apply single and multiple aggregate functions to groups.
# 3. Group by single or multiple columns.
# 
# # Concept Explanation
# When you call `df.groupby('column')`, pandas doesn't calculate anything immediately. It returns a `DataFrameGroupBy` object holding the grouping metadata. You then chain an aggregation method (like `.mean()`, `.sum()`, or `.agg()`) to apply the calculation to those groups.
# 
# # Beginner Examples

# %%
import pandas as pd

data = {
    'store': ['A', 'A', 'B', 'B', 'C', 'C'],
    'product_type': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Clothing'],
    'sales': [1000, 500, 1200, 800, 900, 300]
}
df = pd.DataFrame(data)
print("Original Sales Data:\n", df)

# Group by a single column and sum the sales
# Note: we select ['sales'] to avoid summing strings
store_sales = df.groupby('store')['sales'].sum()
print("\nTotal Sales by Store:\n", store_sales)

# Getting the mean sales per store
store_avg = df.groupby('store')['sales'].mean()
print("\nAverage Sales by Store:\n", store_avg)

# %% [markdown]
# # Intermediate Examples

# %%
# Grouping by multiple columns
store_product_sales = df.groupby(['store', 'product_type'])['sales'].sum()
print("Sales by Store and Product Type:\n", store_product_sales)

# Using .agg() to apply multiple functions at once
multi_agg = df.groupby('store')['sales'].agg(['sum', 'mean', 'count'])
print("\nMultiple Aggregations:\n", multi_agg)

# The result of groupby is often a MultiIndex. 
# Resetting the index turns it back into a standard DataFrame
flattened_groupby = store_product_sales.reset_index()
print("\nFlattened Groupby Output:\n", flattened_groupby)

# Iterating through groups
print("\nIterating through groups:")
for name, group_df in df.groupby('store'):
    print(f"Group: {name}")
    print(group_df)
    print("-" * 15)

# %% [markdown]
# # Machine Learning Relevance
# `groupby` is heavily used in Feature Engineering. For instance, creating a feature like "historical average spend of user" requires grouping the dataset by user_id and taking the mean of their spend. Or, you might calculate "number of transactions in the last week per store". These aggregate statistics become highly predictive features for ML models.
# 
# # Common Mistakes
# - Forgetting to specify the column(s) you want to aggregate after the `groupby`, resulting in Pandas trying to aggregate every column (which throws errors on string columns in newer pandas versions).
# - Getting confused by the MultiIndex output when grouping by multiple columns. Use `.reset_index()` to flatten it back to a normal DataFrame.
# 
# # Interview Questions
# 1. Explain the "split-apply-combine" pattern in the context of pandas groupby.
# 2. What does `df.groupby('col')` return before an aggregation function is applied?
# 3. How do you group by multiple columns?
# 4. How can you apply different aggregation functions to different columns in the same groupby operation?
# 5. How do you convert a groupby result with a MultiIndex back into a flat DataFrame?
# 
# # Practice Problems
# 1. Create a DataFrame `transactions` with columns: `user_id`, `category`, and `amount`.
# 2. Group by `category` and find the total `amount` spent in each category.
# 3. Group by `user_id` and find the total number of transactions (count) and the average amount per user.
# 4. Group by both `user_id` and `category` to find the max transaction amount for each user in each category.
# 5. Reset the index of the result from Problem 4.
# 
# # Solutions

# %%
# Problem 1
tx_data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 2, 3],
    'category': ['Food', 'Gas', 'Food', 'Gas', 'Food', 'Gas'],
    'amount': [25, 40, 30, 50, 15, 60]
})

# Problem 2
cat_sum = tx_data.groupby('category')['amount'].sum()
print("Total by Category:\n", cat_sum)

# Problem 3
user_stats = tx_data.groupby('user_id')['amount'].agg(['count', 'mean'])
print("\nUser Stats (Count, Mean):\n", user_stats)

# Problem 4
user_cat_max = tx_data.groupby(['user_id', 'category'])['amount'].max()
print("\nMax Amount by User/Category:\n", user_cat_max)

# Problem 5
flat_result = user_cat_max.reset_index()
print("\nFlattened Result:\n", flat_result)

# %% [markdown]
# # Further Reading
# - Pandas GroupBy: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
