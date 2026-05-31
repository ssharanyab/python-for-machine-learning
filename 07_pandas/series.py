# %% [markdown]
# # Pandas Series
# 
# # Why this matters
# A Series is the fundamental building block of pandas—a one-dimensional labeled array capable of holding any data type. Understanding Series is crucial because a DataFrame (the primary data structure in pandas) is simply a collection of Series sharing an index. Grasping this concept makes data manipulation significantly easier in machine learning.
# 
# # Learning Objectives
# 1. Understand what a pandas Series is and how it differs from a NumPy array.
# 2. Learn how to create, access, and manipulate data within a Series.
# 3. Understand index alignment and vectorized operations on Series.
# 
# # Concept Explanation
# Think of a pandas Series as a column in an Excel spreadsheet. It consists of two main components:
# 1. An array of actual data (which can be integers, strings, floats, objects, etc.).
# 2. An associated array of data labels, called its *index*.
# 
# # Beginner Examples
# Let's start by creating some basic Series.

# %%
import pandas as pd
import numpy as np

# Creating a Series from a list
sales_data = [150.5, 200.0, 350.25, 400.1]
sales_series = pd.Series(sales_data)
print("Basic Series:\n", sales_series)

# Creating a Series with a custom index
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
sales_with_index = pd.Series(sales_data, index=days)
print("\nSeries with custom index:\n", sales_with_index)

# Creating from a dictionary
customer_counts = {'Mon': 50, 'Tue': 65, 'Wed': 80}
customer_series = pd.Series(customer_counts)
print("\nSeries from dictionary:\n", customer_series)

# %% [markdown]
# # Intermediate Examples
# Now let's explore indexing, slicing, and vectorized operations on Series.

# %%
# Accessing elements
print("Sales on Tuesday:", sales_with_index['Tuesday'])
print("First two days of sales:\n", sales_with_index[:2])

# Vectorized Operations
# E.g., Applying a 10% discount to all sales
discounted_sales = sales_with_index * 0.9
print("\nDiscounted Sales:\n", discounted_sales)

# Filtering with boolean indexing
high_sales = sales_with_index[sales_with_index > 250]
print("\nSales > 250:\n", high_sales)

# Dealing with missing values in arithmetic operations
series1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
series2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])
print("\nAddition with misaligned indices:\n", series1 + series2)

# %% [markdown]
# # Machine Learning Relevance
# In machine learning, a Series often represents a single feature (variable) or the target variable (labels). For example, a target column `y` extracted from a DataFrame `df['target']` is a Series. When dealing with time series forecasting, a pandas Series indexed by datetime is the standard input format for many algorithms (like ARIMA or Prophet).
# 
# # Common Mistakes
# - Confusing integer-based positional indexing with label-based indexing (e.g., using `s[0]` when the index label is also an integer but doesn't start at 0). Use `.iloc` and `.loc` to be explicit.
# - Modifying a Series inplace accidentally or expecting operations to happen inplace when they return a new Series by default.
# 
# # Interview Questions
# 1. What is the difference between a pandas Series and a NumPy 1D array?
# 2. How does pandas handle operations between two Series with different indices?
# 3. What is the difference between `.loc` and `.iloc` when accessing elements in a Series?
# 4. How can you find the unique values and their frequencies in a pandas Series?
# 5. How do you convert a Series to a Python list or a NumPy array?
# 
# # Practice Problems
# 1. Create a Series of 5 random integers representing daily website visits, indexed by dates from '2023-01-01' to '2023-01-05'.
# 2. Extract the visits for '2023-01-03' from the Series.
# 3. Calculate the total visits and the average visits over the 5 days.
# 4. Identify the dates where visits were greater than the average.
# 5. Multiply the visits by 1.5 to simulate a marketing campaign effect, rounding to the nearest integer.
# 
# # Solutions

# %%
# Problem 1
dates = pd.date_range(start='2023-01-01', periods=5)
visits = pd.Series(np.random.randint(100, 1000, size=5), index=dates)
print("Visits Series:\n", visits)

# Problem 2
print("\nVisits on Jan 3rd:", visits['2023-01-03'])

# Problem 3
print("\nTotal Visits:", visits.sum())
print("Average Visits:", visits.mean())

# Problem 4
above_avg = visits[visits > visits.mean()]
print("\nDays with above average visits:\n", above_avg)

# Problem 5
projected = (visits * 1.5).round().astype(int)
print("\nProjected visits:\n", projected)

# %% [markdown]
# # Further Reading
# - Pandas Official Documentation on Series: https://pandas.pydata.org/docs/reference/series.html
# - Python Data Science Handbook by Jake VanderPlas
