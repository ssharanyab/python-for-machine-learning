# %% [markdown]
# # Pandas DataFrames
# 
# # Why this matters
# The DataFrame is the absolute workhorse of data science and machine learning in Python. It provides a 2-dimensional tabular data structure with labeled axes (rows and columns). Most datasets you'll work with in ML are loaded into, cleaned within, and exported from DataFrames.
# 
# # Learning Objectives
# 1. Understand how to create DataFrames from various data structures.
# 2. Learn basic DataFrame inspection methods.
# 3. Master selecting, adding, and deleting columns.
# 
# # Concept Explanation
# A DataFrame is akin to a table in a relational database or a sheet in Excel. It is a dictionary-like container for Series objects. Every column in a DataFrame is a Series, and they all share the same row index.
# 
# # Beginner Examples

# %%
import pandas as pd
import numpy as np

# Creating DataFrame from a dictionary of lists
data = {
    'product_id': [101, 102, 103, 104],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Home'],
    'price': [299.99, 45.50, 899.00, 15.99]
}
df = pd.DataFrame(data)
print("DataFrame from dict:\n", df)

# Inspecting the DataFrame
print("\nShape of df:", df.shape)
print("\nColumns:", df.columns)
print("\nData Types:\n", df.dtypes)

# Accessing a specific column (Returns a Series)
prices = df['price']
print("\nPrice Column:\n", prices)

# %% [markdown]
# # Intermediate Examples

# %%
# Adding a new column
df['in_stock'] = [True, False, True, True]
print("DataFrame with new column:\n", df)

# Creating a derived column
df['discount_price'] = df['price'] * 0.9
print("\nDataFrame with derived column:\n", df)

# Deleting a column
df = df.drop('discount_price', axis=1) # axis=1 specifies column

# Setting a column as the index
df_indexed = df.set_index('product_id')
print("\nDataFrame with custom index:\n", df_indexed)

# Resetting the index
df_reset = df_indexed.reset_index()

# Basic descriptive statistics
print("\nDescribe:\n", df.describe(include='all'))

# %% [markdown]
# # Machine Learning Relevance
# Scikit-learn (the primary ML library in Python) heavily integrates with pandas. While it inherently uses NumPy arrays for computation, recent versions natively support inputting pandas DataFrames and can output them in certain preprocessing steps (like keeping feature names). A DataFrame holds your feature matrix (X) and target vector (y). Exploring the DataFrame (`.describe()`, `.info()`) is the first step in Exploratory Data Analysis (EDA) before building any ML model.
# 
# # Common Mistakes
# - Using dot notation (e.g., `df.price`) to create a new column instead of bracket notation (`df['new_price']`). Dot notation only works for reading existing columns that have valid Python identifier names.
# - Confusing `axis=0` (rows) and `axis=1` (columns) when dropping data or applying functions.
# 
# # Interview Questions
# 1. What is a pandas DataFrame?
# 2. How do you view the first and last 10 rows of a DataFrame?
# 3. What is the difference between dropping a column with `del df['col']` vs `df.drop('col', axis=1)`?
# 4. How would you change the name of a specific column in a DataFrame?
# 5. Explain how a DataFrame represents data in memory differently than a 2D Python list.
# 
# # Practice Problems
# 1. Create a DataFrame `customers` with columns: `customer_id`, `age`, `signup_date`. Include 4 rows of fictional data.
# 2. Add a new column `is_millennial` which is True if age is between 25 and 40, False otherwise.
# 3. Rename the column `signup_date` to `join_date`.
# 4. Display a summary of the numerical columns using `.describe()`.
# 5. Extract the `age` column as a NumPy array.
# 
# # Solutions

# %%
# Problem 1
cust_data = {
    'customer_id': ['C1', 'C2', 'C3', 'C4'],
    'age': [22, 35, 45, 28],
    'signup_date': ['2023-01-15', '2022-11-20', '2023-03-10', '2021-07-05']
}
customers = pd.DataFrame(cust_data)
print("Customers DataFrame:\n", customers)

# Problem 2
customers['is_millennial'] = (customers['age'] >= 25) & (customers['age'] <= 40)
print("\nWith is_millennial:\n", customers)

# Problem 3
customers = customers.rename(columns={'signup_date': 'join_date'})
print("\nRenamed column:\n", customers)

# Problem 4
print("\nSummary Stats:\n", customers.describe())

# Problem 5
age_array = customers['age'].values
print("\nAge array:\n", age_array, type(age_array))

# %% [markdown]
# # Further Reading
# - Pandas Official Documentation on DataFrame: https://pandas.pydata.org/docs/reference/frame.html
