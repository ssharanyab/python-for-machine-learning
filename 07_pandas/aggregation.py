# %% [markdown]
# # Advanced Aggregation
# 
# # Why this matters
# While basic grouping provides sums and means, real-world data science requires more complex, bespoke aggregations. Mastering `.agg()` and custom aggregation functions allows you to extract highly specific insights and features from your datasets.
# 
# # Learning Objectives
# 1. Apply different aggregation functions to different columns simultaneously.
# 2. Use custom Python functions for aggregation.
# 3. Understand named aggregations for cleaner output.
# 
# # Concept Explanation
# The `.agg()` (or `.aggregate()`) method in pandas allows you to specify exactly which statistical operations to apply, and to which columns, often by passing a dictionary mapping columns to functions.
# 
# # Beginner Examples

# %%
import pandas as pd
import numpy as np

data = {
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering', 'HR'],
    'salary': [60000, 80000, 100000, 120000, 75000],
    'years_experience': [2, 5, 4, 8, 3]
}
df = pd.DataFrame(data)
print("Employee Data:\n", df)

# Applying different aggregations to different columns
agg_dict = {
    'salary': ['mean', 'max'],
    'years_experience': 'mean'
}
dept_stats = df.groupby('department').agg(agg_dict)
print("\nDepartment Stats:\n", dept_stats)

# %% [markdown]
# # Intermediate Examples

# %%
# Named Aggregations (Cleaner approach introduced in newer pandas versions)
# This avoids multi-level column indexes which are hard to work with
clean_agg = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    avg_exp=('years_experience', 'mean')
)
print("Named Aggregations:\n", clean_agg)

# Custom Aggregation Function
def salary_range(x):
    return x.max() - x.min()

custom_agg = df.groupby('department').agg(
    salary_spread=('salary', salary_range)
)
print("\nCustom Aggregation (Spread):\n", custom_agg)

# Using lambda functions
lambda_agg = df.groupby('department').agg(
    salary_range_lambda=('salary', lambda x: x.max() - x.min())
)
print("\nLambda Aggregation:\n", lambda_agg)

# %% [markdown]
# # Machine Learning Relevance
# Complex aggregations generate custom features. For example, capturing the "spread" (max - min) of a user's transaction amounts over the last 30 days might indicate volatility, which could be a strong predictor for churn or fraud models. Named aggregations make generating these features pipeline-ready because you immediately get clean column names to feed into Scikit-Learn.
# 
# # Common Mistakes
# - Relying heavily on `apply` for aggregations when a vectorized numpy or pandas function exists (e.g., writing a custom sum function instead of using `.sum()`). Custom Python functions inside `agg()` or `apply()` are slow.
# - Failing to handle the MultiIndex columns created by the dictionary approach, which makes selecting columns later difficult (e.g., trying to access `df['salary']` when it is now `df[('salary', 'mean')]`).
# 
# # Interview Questions
# 1. How do you calculate the mean for column A and the sum for column B in a single groupby operation?
# 2. What is a "Named Aggregation" in pandas?
# 3. Why might you prefer a named aggregation over passing a dictionary to `.agg()`?
# 4. How do you use a custom function (or lambda) inside `.agg()`?
# 5. What are the performance implications of using custom functions in `.agg()` compared to built-in string methods like 'sum' or 'mean'?
# 
# # Practice Problems
# 1. Create a DataFrame `orders` with `customer_id`, `order_total`, and `items_count`.
# 2. Group by `customer_id` and use `.agg()` with a dictionary to find the sum of `order_total` and the max of `items_count`.
# 3. Use named aggregations to achieve the same result as Problem 2, naming the columns `total_spent` and `max_items_in_order`.
# 4. Write a custom function that returns the 90th percentile of a series.
# 5. Apply your custom function to calculate the 90th percentile of `order_total` for each customer using `.agg()`.
# 
# # Solutions

# %%
# Problem 1
orders = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2],
    'order_total': [100, 150, 50, 200, 300],
    'items_count': [2, 3, 1, 4, 5]
})

# Problem 2
dict_agg = orders.groupby('customer_id').agg({
    'order_total': 'sum',
    'items_count': 'max'
})
print("Dict Aggregation:\n", dict_agg)

# Problem 3
named_agg = orders.groupby('customer_id').agg(
    total_spent=('order_total', 'sum'),
    max_items_in_order=('items_count', 'max')
)
print("\nNamed Aggregations:\n", named_agg)

# Problem 4
def percentile_90(x):
    return x.quantile(0.90)

# Problem 5
p90_agg = orders.groupby('customer_id').agg(
    total_spent_90p=('order_total', percentile_90)
)
print("\n90th Percentile Aggregation:\n", p90_agg)

# %% [markdown]
# # Further Reading
# - Pandas Named Aggregation: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#named-aggregation
