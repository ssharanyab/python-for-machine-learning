# %% [markdown]
# # Array Indexing: Extracting Data Like a Pro
# 
# # Why this matters
# In ML, you rarely process an entire dataset blindly. You might need to extract only the labels from a dataset, pull out the red channel of an image, or filter out specific outliers. NumPy provides powerful, fast indexing methods (including boolean indexing) that allow you to query data without writing slow, cumbersome Python `for` loops.
# 
# # Learning Objectives
# 1. Master basic indexing for 1D, 2D, and N-dimensional arrays.
# 2. Use boolean indexing (masking) to filter data based on conditions.
# 3. Use integer array indexing (fancy indexing) to select specific elements.
# 
# # Concept Explanation
# - **Basic Indexing:** Similar to Python lists. Uses zero-based indexing `[i]`. For 2D, it's `[row, col]`.
# - **Boolean Indexing (Masking):** Passing an array of `True`/`False` values to an array. Only the elements corresponding to `True` are returned. This is the standard way to filter data.
# - **Fancy Indexing:** Passing arrays of integers to access multiple specific elements at once.
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: 1D Indexing
features = np.array([10, 20, 30, 40, 50])
print("First element:", features[0])
print("Last element:", features[-1])

# Example 2: 2D Indexing (Tabular Data)
# 3 samples, 3 features
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# Get element at row 1, col 2 (Zero-indexed)
print("\nElement at (1, 2):", data[1, 2]) # 6

# Example 3: Boolean Indexing (Masking)
# Finding all features greater than 25
mask = features > 25
print("\nBoolean mask:", mask)
filtered_features = features[mask] # Or just features[features > 25]
print("Filtered features:", filtered_features)

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 4: Complex Boolean Indexing
# Using logical AND (&), OR (|) operators
# Find features between 15 AND 45
mask_complex = (features > 15) & (features < 45) # Parentheses are REQUIRED!
print("Complex filtered:", features[mask_complex])

# Example 5: Modifying data using boolean indexing
# Setting all negative values in an array to 0 (like a ReLU activation function!)
activations = np.array([-1.5, 2.0, -0.5, 3.1])
activations[activations < 0] = 0
print("\nReLU output:", activations)

# Example 6: Fancy Indexing
# Selecting specific rows from a dataset
indices_to_select = [0, 2] # Select 1st and 3rd sample
subset = data[indices_to_select]
print("\nSubset of data:\n", subset)

# Example 7: Indexing an Image
# Suppose we have a 3x3 RGB image (3x3x3 tensor)
image = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
# Get the pixel value at row 0, col 0, channel 1 (Green)
green_pixel = image[0, 0, 1]
print("\nGreen pixel at (0,0):", green_pixel)

# %% [markdown]
# # Machine Learning Relevance
# Filtering data is a daily task in Data Science. For example, in a dataset of house prices, you might want to filter out anomalies: `clean_data = data[data[:, 'price'] < 1000000]`. In neural networks, masking is used heavily (e.g., masking padding tokens in NLP so they don't affect attention scores). Modifying values with masks (like our ReLU example) is how piecewise functions are implemented efficiently.
# 
# # Common Mistakes
# 1. **Missing Parentheses in Complex Masks:** Doing `features > 15 & features < 45` will raise a `ValueError`. Python's bitwise `&` has high precedence, so it evaluates `15 & features` first. Always use `(features > 15) & (features < 45)`.
# 2. **Using `and` instead of `&`:** The Python `and` keyword evaluates truthiness of the *entire array*, which is ambiguous. You must use the bitwise operator `&` for element-wise logical AND in NumPy.
# 3. **Copy vs View:** Boolean and Fancy indexing always return a *copy* of the data. Modifying the copy will not change the original array.
# 
# # Interview Questions
# 1. **How do you replace all values in a NumPy array `arr` that are greater than 10 with the value 10?**
#    *Answer:* `arr[arr > 10] = 10`. (This is also known as clipping/clamping).
# 2. **Why does `arr[(arr > 5) and (arr < 10)]` raise an error?**
#    *Answer:* Because the `and` keyword tries to evaluate the truth value of an entire boolean array, which is ambiguous. You must use the element-wise bitwise operator `&` instead.
# 3. **What is Fancy Indexing in NumPy?**
#    *Answer:* Fancy indexing refers to passing arrays of indices in place of single scalars to access multiple array elements at once.
# 4. **If you use boolean indexing to create `arr2 = arr[arr > 0]`, does changing `arr2` affect `arr`?**
#    *Answer:* No, boolean indexing always returns a copy of the data, not a view.
# 5. **Given a 2D array, how do you extract elements at coordinates (0,0), (1,1), and (2,2)?**
#    *Answer:* Using fancy indexing: `arr[[0, 1, 2], [0, 1, 2]]`.
# 
# # Practice Problems
# 1. Create an array of numbers from 1 to 10. Extract all even numbers.
# 2. Given a 2D array of shape (4, 4) with numbers 0-15, print the element in the 3rd row and 4th column.
# 3. Create an array of random normal floats. Clip all values below -1 to -1, and all values above 1 to 1.
# 4. Create an array `[10, 20, 30, 40, 50]`. Use fancy indexing to extract the 2nd, 3rd, and 5th elements.
# 5. Given a 2D array, extract the diagonal elements using fancy indexing.
# 
# # Solutions
# 
# %%
# Solution 1
arr = np.arange(1, 11)
evens = arr[arr % 2 == 0]
print("Q1 evens:", evens)

# Solution 2
mat = np.arange(16).reshape(4, 4)
print("Q2 element:", mat[2, 3]) # 3rd row (index 2), 4th col (index 3)

# Solution 3
normals = np.random.randn(5)
normals[normals < -1] = -1
normals[normals > 1] = 1
print("Q3 clipped:", normals)

# Solution 4
arr4 = np.array([10, 20, 30, 40, 50])
subset = arr4[[1, 2, 4]]
print("Q4 subset:", subset)

# Solution 5
mat5 = np.arange(9).reshape(3, 3)
diag = mat5[[0, 1, 2], [0, 1, 2]]
print("Q5 diagonal:", diag)

# %% [markdown]
# # Further Reading
# - NumPy Indexing: https://numpy.org/doc/stable/user/basics.indexing.html
