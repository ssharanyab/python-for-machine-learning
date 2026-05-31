# %% [markdown]
# # NumPy Reshaping
# 
# # Why this matters
# Data in Machine Learning often needs to be reshaped to fit into models. For instance, images loaded as 1D arrays might need to be reshaped into 2D or 3D arrays (height, width, channels) for Convolutional Neural Networks, or tabular data might need an extra dimension to be processed by a recurrent neural network.
# 
# # Learning Objectives
# - Understand the shape attribute of NumPy arrays.
# - Learn how to use `reshape()` to change array dimensions without changing data.
# - Understand flattening (`flatten()`, `ravel()`) and adding axes (`np.newaxis`, `expand_dims()`).
# - Apply reshaping to images, tabular data, and feature vectors.
# 
# # Concept Explanation
# Reshaping changes the way data is indexed, but the underlying data remains the same in memory.
# - `reshape(shape)`: Gives a new shape to an array.
# - `flatten()`: Returns a copy of the array collapsed into one dimension.
# - `ravel()`: Returns a contiguous flattened array (a view whenever possible).
# 
# # Beginner Examples
# %%
import numpy as np

# Example 1: 1D to 2D
arr1 = np.arange(1, 13)
reshaped_2d = arr1.reshape(3, 4)
print("1D to 2D:\n", reshaped_2d)

# Example 2: 2D to 1D (Flattening)
flattened = reshaped_2d.flatten()
print("\nFlattened:\n", flattened)

# Example 3: Using -1 for unknown dimensions
reshaped_auto = arr1.reshape(2, -1)
print("\nReshaped with -1 (2 rows):\n", reshaped_auto)

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Images (3D to 2D for traditional ML)
# Imagine 10 grayscale images of 28x28 pixels
images = np.random.rand(10, 28, 28)
# Reshape to 10 samples, 784 features
tabular_images = images.reshape(10, -1)
print(f"\nOriginal image shape: {images.shape}, Tabular shape: {tabular_images.shape}")

# Example 5: Tabular to Sequence (2D to 3D for RNNs)
# 100 samples, 20 time steps, 1 feature
tabular = np.random.rand(100, 20)
sequence = np.expand_dims(tabular, axis=-1)
print(f"Tabular shape: {tabular.shape}, Sequence shape: {sequence.shape}")

# Example 6: Feature Vector Broadcasting
feature_vector = np.array([1, 2, 3])
# Convert (3,) to (3, 1) to broadcast over a (3, 4) matrix
column_vector = feature_vector[:, np.newaxis]
print(f"\nColumn vector shape: {column_vector.shape}")

# %% [markdown]
# # Machine Learning Relevance
# In Deep Learning, PyTorch and TensorFlow expect specific input shapes. A CNN expects `(batch_size, channels, height, width)` or `(batch_size, height, width, channels)`. `reshape` and `transpose` are essential for this.
# 
# # Common Mistakes
# - Using incompatible dimensions (e.g., reshaping 10 elements into a 3x3 array).
# - Confusing `flatten()` (returns a copy) with `ravel()` (returns a view).
# - Misunderstanding memory layout (C-order vs F-order) when reshaping.
# 
# # Interview Questions
# 1. What is the difference between `flatten()` and `ravel()`?
# 2. How does `reshape(-1, 1)` change a 1D array?
# 3. Can you reshape an array in place?
# 4. What happens if the new shape does not match the total number of elements?
# 5. How do you add an empty dimension to an existing array?
# 
# # Practice Problems
# 1. Create a 1D array of numbers 1-20 and reshape it into a 4x5 matrix.
# 2. Flatten the previous matrix using `ravel()` and modify the first element. Check if the original matrix changed.
# 3. You have a dataset of shape `(50, 10, 10, 3)` representing 50 color images. Reshape it to `(50, 300)`.
# 4. Add a new dimension to a 1D array of size 5 to make it shape `(1, 5)`.
# 5. Reshape a `(6, 4)` array to `(2, 3, 4)` without explicitly specifying all dimensions.
# 
# # Solutions
# %%
# Solution 1
p1 = np.arange(1, 21).reshape(4, 5)
print("P1:\n", p1)

# Solution 2
r_view = p1.ravel()
r_view[0] = 999
print("P2 changed original?", p1[0, 0] == 999)

# Solution 3
images_4d = np.random.rand(50, 10, 10, 3)
flat_features = images_4d.reshape(50, -1)
print("P3 shape:", flat_features.shape)

# Solution 4
arr_1d = np.arange(5)
row_vector = arr_1d[np.newaxis, :]
print("P4 shape:", row_vector.shape)

# Solution 5
arr_6x4 = np.zeros((6, 4))
arr_2x3x4 = arr_6x4.reshape(2, -1, 4)
print("P5 shape:", arr_2x3x4.shape)

# %% [markdown]
# # Further Reading
# - NumPy Documentation: `numpy.reshape`, `numpy.ravel`, `numpy.expand_dims`.
