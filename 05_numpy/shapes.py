# %% [markdown]
# # Array Shapes: Structure of Machine Learning Data
# 
# # Why this matters
# In machine learning, "Shape Errors" (e.g., `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`) are the most common bugs you will encounter. Understanding how data is shaped and how to reshape it is non-negotiable. For instance, a convolutional neural network expects an image shape of `(batch_size, channels, height, width)`, but your data might be loaded as `(num_images, height, width, channels)`. You must know how to transform shapes properly.
# 
# # Learning Objectives
# 1. Understand what the `.shape` attribute represents.
# 2. Learn how to reshape arrays using `.reshape()`.
# 3. Flatten multidimensional arrays using `.flatten()` or `.ravel()`.
# 4. Understand how -1 acts as an inferred dimension in reshaping.
# 5. Transpose arrays.
# 
# # Concept Explanation
# The `shape` of an array is a tuple indicating the number of elements in each dimension. 
# For example, a 2D array with 3 rows and 4 columns has a shape of `(3, 4)`.
# `reshape()` allows you to change the dimensions of an array without changing its data, provided the total number of elements remains the same.
# Transposing swaps the dimensions of an array.
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: Checking Shape
# A 1D feature vector
features = np.array([1.5, 2.3, 0.9, 5.1])
print("Features shape:", features.shape) # (4,) - Note the comma, it's a tuple of 1 element

# Example 2: 2D Shape (Tabular Data)
# 3 samples (rows), 4 features (columns)
tabular_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("Tabular data shape:", tabular_data.shape) # (3, 4)

# Example 3: Reshaping 1D to 2D
# Often Scikit-Learn requires a 2D array even for a single feature
single_feature = np.array([1, 2, 3, 4, 5, 6])
reshaped_data = single_feature.reshape(3, 2)
print("\nReshaped to (3, 2):\n", reshaped_data)

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 4: Using -1 to infer a dimension
# If we have 12 elements and we want 3 rows, we can let NumPy calculate the columns
inferred_shape = single_feature.reshape(3, -1)
print("Inferred shape using (3, -1):\n", inferred_shape)
print("Resulting shape:", inferred_shape.shape)

# Example 5: Flattening an image
# A 3x3 grayscale image (2D) needs to be passed to a Dense Neural Network as a 1D vector
image_3x3 = np.array([
    [255, 128, 0],
    [0, 255, 128],
    [128, 0, 255]
])
flattened = image_3x3.flatten()
print("\nFlattened image shape:", flattened.shape)
print("Flattened image:", flattened)

# Example 6: Transposing a matrix
# Swapping rows and columns
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print("\nOriginal Matrix (3x2):\n", matrix)
print("Transposed Matrix (2x3):\n", matrix.T)

# Example 7: Adding a dummy dimension (Crucial for ML)
# Suppose we have an image of shape (28, 28) and a PyTorch model expects (1, 28, 28) for batch size 1
image_28 = np.ones((28, 28))
image_batch = np.expand_dims(image_28, axis=0) # or image_28[np.newaxis, :]
print("\nExpanded shape:", image_batch.shape)

# %% [markdown]
# # Machine Learning Relevance
# **Shape Visualization:**
# Vector: `[v1, v2, v3, v4]` -> Shape `(4,)`
# Matrix / Batch of Vectors: 
# `[[v1, v2],`
#  `[v3, v4],`
#  `[v5, v6]]` -> Shape `(3, 2)`
# 
# You will constantly use `.reshape(-1, 1)` to convert a 1D array of labels `[0, 1, 0]` into a column vector:
# `[[0],`
#  `[1],`
#  `[0]]`
# This is because many algorithms treat `(N,)` differently than `(N, 1)`. `(N, 1)` explicitly means a 2D matrix with 1 column.
# 
# # Common Mistakes
# 1. **Total Element Mismatch:** Trying to reshape an array of 10 elements into `(3, 4)` will throw a `ValueError`. `10 != 3 * 4`.
# 2. **Confusing (N,) and (N, 1):** A shape of `(N,)` is a rank-1 array (just a sequence). `(N, 1)` is a rank-2 array (a matrix). Matrix multiplication behaves very differently depending on which one you use.
# 3. **Using multiple -1s:** You can only use `-1` for ONE dimension in `.reshape()`. `arr.reshape(-1, -1)` will raise an error because NumPy cannot solve for two unknowns.
# 
# # Interview Questions
# 1. **What does the parameter `-1` do in `np.reshape(-1, 1)`?**
#    *Answer:* It tells NumPy to automatically calculate the size of that dimension based on the total number of elements and the other specified dimensions. Here, it creates a 2D array with 1 column and as many rows as needed.
# 2. **What is the difference between `.flatten()` and `.ravel()`?**
#    *Answer:* `.flatten()` always returns a copy of the array, while `.ravel()` returns a view of the original array whenever possible (meaning modifying the raveled array might modify the original).
# 3. **How do you convert a 1D array of shape `(100,)` into a 2D row vector of shape `(1, 100)`?**
#    *Answer:* By using `arr.reshape(1, -1)` or `arr[np.newaxis, :]`.
# 4. **If a batch of 32 RGB images has shape `(32, 28, 28, 3)`, what is the total number of values in this array?**
#    *Answer:* `32 * 28 * 28 * 3 = 75,264`.
# 5. **Why might you transpose tabular data of shape `(samples, features)`?**
#    *Answer:* Some mathematical formulas or specific algorithms (like computing the covariance matrix of features) expect the data matrix to be of shape `(features, samples)`.
# 
# # Practice Problems
# 1. Create an array of 24 elements using `arange` and reshape it into a 3D array of shape `(2, 3, 4)`.
# 2. Take the 3D array from Problem 1 and flatten it back to 1D using `.ravel()`.
# 3. Create a 1D array of 10 elements and reshape it into a column vector (shape `(10, 1)`).
# 4. Create a matrix of shape `(4, 5)` and transpose it. What is its new shape?
# 5. Given an array of shape `(100, 64, 64, 3)` (100 images), reshape it so that each image is flattened into a 1D vector, resulting in a shape of `(100, 12288)`. Use `-1`.
# 
# # Solutions
# 
# %%
# Solution 1
arr_24 = np.arange(24).reshape(2, 3, 4)
print("Q1 shape:", arr_24.shape)

# Solution 2
raveled = arr_24.ravel()
print("Q2 shape:", raveled.shape)

# Solution 3
col_vec = np.arange(10).reshape(-1, 1)
print("Q3 shape:", col_vec.shape)

# Solution 4
mat_45 = np.zeros((4, 5))
transposed = mat_45.T
print("Q4 shape:", transposed.shape)

# Solution 5
images = np.zeros((100, 64, 64, 3))
flattened_images = images.reshape(images.shape[0], -1)
# or flattened_images = images.reshape(100, -1)
print("Q5 shape:", flattened_images.shape)

# %% [markdown]
# # Further Reading
# - NumPy Reshaping: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
# - Broadcasting and Shapes: https://numpy.org/doc/stable/user/basics.broadcasting.html
