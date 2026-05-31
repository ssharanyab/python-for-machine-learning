# %% [markdown]
# # Feature Engineering in Pandas
# 
# # Why this matters
# Feature Engineering is the process of using domain knowledge to create new variables (features) that make machine learning algorithms work better. Better features often beat better algorithms. Pandas is the primary tool used to craft these features from raw data.
# 
# # Learning Objectives
# 1. Create derived columns via mathematical operations.
# 2. Extract features from strings and datetime objects.
# 3. Use `.map()` and `.apply()` for custom feature creation.
# 
# # Concept Explanation
# Creating a new feature usually involves performing element-wise operations on one or more existing columns and assigning the result to a new column.
# 
# # Beginner Examples

# %%
import pandas as pd
import numpy as np

data = {
    'user_id': [1, 2, 3],
    'purchase_amount': [150.0, 50.0, 300.0],
    'tax_amount': [15.0, 5.0, 30.0],
    'date': ['2023-01-15', '2023-06-20', '2023-12-05'],
    'category': ['electronics', 'clothing', 'electronics']
}
df = pd.DataFrame(data)

# 1. Mathematical Feature Extraction
df['total_amount'] = df['purchase_amount'] + df['tax_amount']
print("Mathematical Feature:\n", df[['user_id', 'total_amount']])

# 2. Categorical Mapping (Encoding)
# Machine Learning models require numbers, not strings.
category_map = {'electronics': 1, 'clothing': 0}
df['category_code'] = df['category'].map(category_map)
print("\nMapped Categorical Feature:\n", df[['category', 'category_code']])

# %% [markdown]
# # Intermediate Examples

# %%
# 3. Datetime Feature Extraction
# Convert string to datetime first
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek # 0=Monday, 6=Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print("Datetime Features:\n", df[['date', 'month', 'day_of_week', 'is_weekend']])

# 4. String Extraction
names_df = pd.DataFrame({'name': ['Smith, John', 'Doe, Jane', 'Dr. Strange']})
# Extracting the last name (text before the comma)
names_df['last_name'] = names_df['name'].str.split(',').str[0]
print("\nString Feature Extraction:\n", names_df)

# 5. Using Apply for complex logic
def categorize_spend(amount):
    if amount > 200:
        return 'High'
    elif amount > 100:
        return 'Medium'
    else:
        return 'Low'

df['spend_level'] = df['purchase_amount'].apply(categorize_spend)
print("\nApply Custom Function:\n", df[['purchase_amount', 'spend_level']])

# %% [markdown]
# # Machine Learning Relevance
# Raw data is rarely ready for an ML model. If you pass an raw datetime string to a Random Forest, it will crash. By extracting the 'Month' and 'Day of Week', you give the model seasonal information. `map()` is a manual way to do Ordinal Encoding, which is formalized later in Scikit-Learn pipelines.
# 
# # Common Mistakes
# - Using `.apply()` for simple mathematical operations. `.apply()` runs a Python loop under the hood and is very slow on large datasets. Always prefer vectorized operations (like `df['A'] + df['B']`) when possible.
# - Forgetting to convert a column to a proper dtype (like datetime) before trying to extract properties (like `.dt.month`), resulting in an AttributeError.
# 
# # Interview Questions
# 1. Why is feature engineering important for machine learning?
# 2. How do you extract the year from a datetime column in pandas?
# 3. What is the difference between `.map()` and `.apply()` on a Series?
# 4. Why should you avoid using `.apply()` if a vectorized alternative exists?
# 5. How would you convert a boolean column (True/False) to integers (1/0)?
# 
# # Practice Problems
# 1. Create a DataFrame with columns `weight_kg` and `height_m`.
# 2. Create a new feature `bmi` calculated as weight / (height^2).
# 3. Create a DataFrame with a `text` column containing sentences. Use `.str.len()` to create a `text_length` feature.
# 4. Map a column `status` containing 'Active', 'Inactive', 'Pending' to integers 1, 0, -1.
# 5. Convert a string date column to datetime and extract the quarter of the year.
# 
# # Solutions

# %%
# Problem 1 & 2
health_df = pd.DataFrame({'weight_kg': [70, 85, 60], 'height_m': [1.75, 1.80, 1.60]})
health_df['bmi'] = health_df['weight_kg'] / (health_df['height_m'] ** 2)
print("BMI Feature:\n", health_df)

# Problem 3
text_df = pd.DataFrame({'text': ['Hello world', 'Pandas is great', 'ML']})
text_df['text_length'] = text_df['text'].str.len()
print("\nText Length:\n", text_df)

# Problem 4
status_df = pd.DataFrame({'status': ['Active', 'Pending', 'Inactive']})
status_map = {'Active': 1, 'Inactive': 0, 'Pending': -1}
status_df['status_code'] = status_df['status'].map(status_map)
print("\nStatus Map:\n", status_df)

# Problem 5
date_df = pd.DataFrame({'date': ['2023-01-01', '2023-05-01', '2023-09-01']})
date_df['date'] = pd.to_datetime(date_df['date'])
date_df['quarter'] = date_df['date'].dt.quarter
print("\nQuarter Feature:\n", date_df)

# %% [markdown]
# # Further Reading
# - Pandas Working with Text Data: https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html
# - Pandas Time Series/Date functionality: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
