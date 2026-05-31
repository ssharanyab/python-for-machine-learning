# %% [markdown]
# # Reading CSV Files
# 
# # Why this matters
# CSV (Comma Separated Values) is the most ubiquitous data storage format for raw datasets. Before you can train a machine learning model, you must load your data into memory. `pd.read_csv` is the standard tool for parsing these files into pandas DataFrames.
# 
# # Learning Objectives
# 1. Master `pd.read_csv()` for loading data.
# 2. Learn how to handle different delimiters, missing values, and date parsing during load.
# 3. Understand how to write a DataFrame back to a CSV.
# 
# # Concept Explanation
# `read_csv` reads a comma-separated values (csv) file into DataFrame. It comes with dozens of parameters to handle almost any quirk in raw data files (e.g., custom separators, skipping rows, specifying data types to save memory).
# 
# # Beginner Examples
# Note: For these examples, we will simulate reading a CSV using Python's `io.StringIO` which acts like a file in memory. In reality, you'd pass a file path like `pd.read_csv('../datasets/sales.csv')`.

# %%
import pandas as pd
import io

csv_data = """id,date,amount,status
1,2023-01-01,150.0,completed
2,2023-01-02,200.5,pending
3,2023-01-02,15.0,completed
"""

# Reading a basic CSV
df = pd.read_csv(io.StringIO(csv_data))
print("Basic read:\n", df)

# Check data types
print("\nData Types:\n", df.dtypes)

# %% [markdown]
# # Intermediate Examples

# %%
csv_messy = """id|date|amount|status
1|2023-01-01|150.0|completed
# this is a comment
2|2023-01-02|N/A|pending
3|2023-01-02|15.0|completed
"""

# Reading with custom delimiter, parsing dates, handling NA, and skipping comments
df_clean = pd.read_csv(
    io.StringIO(csv_messy),
    sep='|',                 # Custom delimiter
    parse_dates=['date'],    # Automatically convert to datetime
    na_values=['N/A'],       # Treat 'N/A' as NaN
    comment='#'              # Ignore comment lines
)

print("\nCleaned DataFrame:\n", df_clean)
print("\nCleaned Data Types:\n", df_clean.dtypes)

# Setting an index column during read
df_indexed = pd.read_csv(io.StringIO(csv_data), index_col='id')
print("\nIndexed DataFrame:\n", df_indexed)

# Saving to CSV (uncomment to run)
# df_clean.to_csv('cleaned_sales.csv', index=False) 

# %% [markdown]
# # Machine Learning Relevance
# The parameters in `read_csv` are often the first step in data preprocessing. By properly defining `na_values`, you save time replacing garbage strings later. By specifying `dtype` for categorical features, you save immense amounts of memory (RAM) which is crucial when loading large datasets for ML training.
# 
# # Common Mistakes
# - Forgetting `index=False` when using `.to_csv()`, resulting in an unnamed column of index numbers being saved and re-loaded later (the infamous `Unnamed: 0` column).
# - Not parsing dates on load, leading to slow string-to-datetime conversions later.
# 
# # Interview Questions
# 1. How do you handle a CSV file that uses tabs instead of commas as separators?
# 2. How can you read only the first 1000 rows of a large CSV file to inspect it?
# 3. What parameter prevents Pandas from saving the row indices when exporting to CSV?
# 4. If a CSV file has missing values represented as "?", how do you handle this during the read process?
# 5. How can you reduce memory usage when reading a large CSV file containing string columns with few unique values?
# 
# # Practice Problems
# 1. Use `io.StringIO` to create a mock CSV string containing `user_id, age, country`. Include one row with a missing age represented as "UNKNOWN".
# 2. Read this mock CSV into a DataFrame, ensuring "UNKNOWN" is parsed as NaN.
# 3. Read the CSV but specify that `user_id` should be parsed as a string (object), not an integer.
# 4. Create a DataFrame and export it to a mock CSV string using `to_csv()` without the index.
# 5. Parse a CSV containing a 'timestamp' column directly into datetime objects.
# 
# # Solutions

# %%
# Problem 1 & 2
mock_csv = """user_id,age,country
101,25,USA
102,UNKNOWN,UK
103,42,Canada
"""
df_users = pd.read_csv(io.StringIO(mock_csv), na_values=['UNKNOWN'])
print("Users DataFrame:\n", df_users)

# Problem 3
df_users_str = pd.read_csv(io.StringIO(mock_csv), dtype={'user_id': str})
print("\nData Types:\n", df_users_str.dtypes)

# Problem 4
output_csv = df_users.to_csv(index=False)
print("\nExported CSV String:\n", output_csv)

# Problem 5
time_csv = """event_id,timestamp
1,2023-05-01 12:00:00
2,2023-05-01 13:30:00"""
df_time = pd.read_csv(io.StringIO(time_csv), parse_dates=['timestamp'])
print("\nTime DataFrame types:\n", df_time.dtypes)

# %% [markdown]
# # Further Reading
# - `read_csv` documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
