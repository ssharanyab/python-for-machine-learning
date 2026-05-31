# %% [markdown]
# # Merging DataFrames
# 
# # Why this matters
# In real-world enterprise databases, data is normalized and spread across multiple tables (e.g., a `customers` table and an `orders` table). To train an ML model, you usually need a single, flat, analytical dataset. `pd.merge()` is how you perform SQL-style JOINs in pandas to combine these tables based on common keys.
# 
# # Learning Objectives
# 1. Understand how `pd.merge()` works.
# 2. Differentiate between inner, outer, left, and right merges.
# 3. Merge on specific columns when names differ.
# 
# # Concept Explanation
# Merging combines DataFrames horizontally, aligning rows based on the values in designated "key" columns.
# 
# # Beginner Examples

# %%
import pandas as pd

customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104],
    'customer_id': [1, 1, 2, 4], # Note: Customer 4 doesn't exist in customers table
    'amount': [250, 150, 300, 50]
})

print("Customers:\n", customers)
print("\nOrders:\n", orders)

# Inner Merge (Default) - Only keeps rows where the key exists in BOTH DataFrames
inner_merge = pd.merge(customers, orders, on='customer_id', how='inner')
print("\nInner Merge:\n", inner_merge)

# %% [markdown]
# # Intermediate Examples

# %%
# Left Merge - Keeps all rows from the Left DataFrame (customers)
left_merge = pd.merge(customers, orders, on='customer_id', how='left')
print("Left Merge:\n", left_merge)
# Notice Charlie has NaN for order_id and amount because he made no orders.

# Right Merge - Keeps all rows from the Right DataFrame (orders)
right_merge = pd.merge(customers, orders, on='customer_id', how='right')
print("\nRight Merge:\n", right_merge)
# Notice Order 104 has NaN for name because Customer 4 isn't in the customers table.

# Outer Merge - Keeps all rows from BOTH DataFrames
outer_merge = pd.merge(customers, orders, on='customer_id', how='outer')
print("\nOuter Merge:\n", outer_merge)

# Merging when column names are different
orders_diff_name = orders.rename(columns={'customer_id': 'c_id'})
diff_name_merge = pd.merge(customers, orders_diff_name, left_on='customer_id', right_on='c_id', how='inner')
print("\nMerge with different column names:\n", diff_name_merge)

# %% [markdown]
# # Machine Learning Relevance
# Merging is the core of dataset construction. If you want to predict if an order will be returned, the `orders` table only has the amount. You must merge it with the `customers` table to get the customer's age, account age, and location. Data leakage can accidentally occur during merges if you perform an inner join that unintentionally drops a class of users from your training set.
# 
# # Common Mistakes
# - Unintentional Cartesian Products (explosion of rows) happening when merging on a column that has duplicate values in BOTH DataFrames. Always check the `.shape` of your DataFrame before and after a merge.
# - Losing rows silently by defaulting to an `inner` join when a `left` join was intended.
# 
# # Interview Questions
# 1. Explain the difference between an INNER JOIN and a LEFT JOIN in pandas.
# 2. What happens if there are duplicate keys in both DataFrames being merged?
# 3. How do you merge two DataFrames if the key column has a different name in each DataFrame?
# 4. After a left merge, how does pandas represent the missing values from the right table?
# 5. How can you verify that a merge didn't accidentally increase the number of rows unexpectedly?
# 
# # Practice Problems
# 1. Create a `products` DataFrame (product_id, product_name) and an `inventory` DataFrame (product_id, quantity).
# 2. Perform an inner merge on `product_id`.
# 3. Create a `sales` DataFrame (item_id, amount). Merge it with `products` assuming `item_id` matches `product_id`, keeping all rows from `sales`.
# 4. What parameter would you use to indicate which DataFrame the columns came from if they have identical names (like `updated_at`)?
# 5. Check the shape of the result from Problem 3.
# 
# # Solutions

# %%
# Problem 1
products = pd.DataFrame({'product_id': [1, 2, 3], 'product_name': ['A', 'B', 'C']})
inventory = pd.DataFrame({'product_id': [1, 2, 4], 'quantity': [10, 20, 30]})

# Problem 2
merged_inv = pd.merge(products, inventory, on='product_id', how='inner')
print("Inner Merge:\n", merged_inv)

# Problem 3
sales = pd.DataFrame({'item_id': [1, 3, 5], 'amount': [100, 200, 300]})
sales_merge = pd.merge(sales, products, left_on='item_id', right_on='product_id', how='left')
print("\nLeft Merge Sales:\n", sales_merge)

# Problem 4
# You use the `suffixes` parameter, e.g., suffixes=('_prod', '_inv')
dup_col_df1 = pd.DataFrame({'id': [1], 'val': ['a']})
dup_col_df2 = pd.DataFrame({'id': [1], 'val': ['b']})
suff_merge = pd.merge(dup_col_df1, dup_col_df2, on='id', suffixes=('_left', '_right'))
print("\nSuffixes:\n", suff_merge)

# Problem 5
print("\nShape of sales_merge:", sales_merge.shape)

# %% [markdown]
# # Further Reading
# - Pandas Merge, Join, Concatenate: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
