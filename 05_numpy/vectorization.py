# %% [markdown]
# # NumPy Vectorization and Broadcasting
# 
# # Why this matters
# In Python, `for` loops are slow. Vectorization allows NumPy to apply operations to entire arrays efficiently in C. Broadcasting allows these operations between arrays of different shapes, which is the backbone of ML operations like adding bias vectors to weight matrices.
# 
# # Learning Objectives
# - Replace loops with vectorized NumPy operations.
# - Understand broadcasting rules.
# - Apply operations to tabular features and image pixels efficiently.
# 
# # Concept Explanation
# - **Vectorization**: Performing operations on arrays rather than iterating element-by-element.
# - **Broadcasting**: How NumPy treats arrays with different shapes during arithmetic operations. The smaller array is "broadcast" across the larger array.
#   - Rule: Dimensions are compatible if they are equal, or if one of them is 1.
# 
# # Beginner Examples
# %%
import numpy as np

# Example 1: Vectorized Math
arr = np.array([1, 2, 3, 4, 5])
print("Squared:\n", arr ** 2)

# Example 2: Scalar Broadcasting
print("Add 10:\n", arr + 10)

# Example 3: Broadcasting between arrays
matrix = np.ones((3, 3))
row = np.array([1, 2, 3])
print("\nMatrix + Row:\n", matrix + row)

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Tabular Data - Feature Scaling (Min-Max Scaling)
X = np.random.rand(100, 5) * 50  # 100 samples, 5 features
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)
print(f"Original X first row: {X[0]}")
print(f"Scaled X first row: {X_scaled[0]}")

# Example 5: Images - Brightness adjustment
image = np.random.randint(0, 255, (256, 256, 3))
# Add 50 to brightness, clip to 255
bright_image = np.clip(image + 50, 0, 255)
print(f"Max pixel value: {bright_image.max()}")

# Example 6: Calculating Distances (Euclidean)
point = np.array([0, 0])
dataset = np.random.rand(10, 2)
distances = np.sqrt(np.sum((dataset - point)**2, axis=1))
print("Distances shape:", distances.shape)

# %% [markdown]
# # Machine Learning Relevance
# Vectorization is mandatory for writing performant ML algorithms from scratch, like Gradient Descent, K-Nearest Neighbors, or Neural Network forward passes. Loop-based implementations will be magnitudes slower.
# 
# # Common Mistakes
# - Broadcasting errors: Attempting to broadcast incompatible shapes (e.g., `(3, 3)` and `(4,)`).
# - Forgetting to use `np.newaxis` to align dimensions properly for broadcasting.
# 
# # Interview Questions
# 1. Explain NumPy's broadcasting rules.
# 2. Why is vectorization faster than Python loops?
# 3. How do you center a dataset matrix `X` by subtracting the mean of each column?
# 4. Will shape `(4, 1, 5)` broadcast with `(3, 5)`?
# 5. What is the output shape when adding a `(100, 64, 64, 3)` image batch and a `(3,)` array?
# 
# # Practice Problems
# 1. Create a `(5, 5)` matrix and subtract the row mean from each row.
# 2. Given a feature matrix `X` of shape `(1000, 10)`, multiply each feature (column) by a specific weight from a weight array `W` of shape `(10,)`.
# 3. Calculate the Mean Squared Error between predicted array `Y_pred` and true array `Y_true`.
# 4. Normalize a grayscale image of shape `(28, 28)` to have values between 0 and 1.
# 5. Add a bias vector `b` of shape `(128,)` to the output of a linear layer of shape `(64, 128)`.
# 
# # Solutions
# %%
# Solution 1
mat = np.random.rand(5, 5)
row_means = mat.mean(axis=1, keepdims=True)
p1 = mat - row_means

# Solution 2
X = np.random.rand(1000, 10)
W = np.random.rand(10)
p2 = X * W  # Broadcasting handles it!

# Solution 3
Y_true = np.array([1, 2, 3])
Y_pred = np.array([1.1, 1.9, 3.2])
mse = np.mean((Y_true - Y_pred)**2)
print("MSE:", mse)

# Solution 4
image = np.random.randint(0, 255, (28, 28))
normalized = image / 255.0

# Solution 5
layer_out = np.random.rand(64, 128)
b = np.random.rand(128)
output = layer_out + b

# %% [markdown]
# # Further Reading
# - NumPy Documentation: Broadcasting rules.
