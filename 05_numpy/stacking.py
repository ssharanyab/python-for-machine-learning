# %% [markdown]
# # NumPy Stacking and Concatenation
# 
# # Why this matters
# In ML, you often need to combine datasets. For instance, combining training and validation sets, adding a new feature column to tabular data, or stacking image channels (R, G, B) to form a color image.
# 
# # Learning Objectives
# - Concatenate arrays along existing axes using `np.concatenate`.
# - Stack arrays along new axes using `np.stack`, `np.vstack`, and `np.hstack`.
# - Split arrays back apart.
# 
# # Concept Explanation
# - `np.concatenate((a1, a2), axis=0)`: Joins arrays along an *existing* axis.
# - `np.vstack((a1, a2))`: Stacks vertically (row-wise).
# - `np.hstack((a1, a2))`: Stacks horizontally (column-wise).
# - `np.stack((a1, a2), axis=0)`: Joins arrays along a *new* axis.
# 
# # Beginner Examples
# %%
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Example 1: Horizontal stack (1D)
h_stack = np.hstack((a, b))
print("HStack:\n", h_stack)

# Example 2: Vertical stack (1D to 2D)
v_stack = np.vstack((a, b))
print("\nVStack:\n", v_stack)

# Example 3: Concatenation
# Concatenate requires same dimensions
c1 = np.array([[1, 2], [3, 4]])
c2 = np.array([[5, 6]])
concat_ax0 = np.concatenate((c1, c2), axis=0)
print("\nConcat axis 0:\n", concat_ax0)

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Tabular Data - Adding Features
# 100 samples, 5 features
X = np.random.rand(100, 5)
# New feature (100 samples, 1 feature)
new_feature = np.random.rand(100, 1)
# Combine them
X_new = np.hstack((X, new_feature))
print(f"Original shape: {X.shape}, New shape: {X_new.shape}")

# Example 5: Images - Stacking color channels
red = np.zeros((32, 32))
green = np.ones((32, 32))
blue = np.zeros((32, 32))
# Stack to form (32, 32, 3)
rgb_image = np.stack((red, green, blue), axis=-1)
print(f"RGB Image Shape: {rgb_image.shape}")

# Example 6: Combining Batches
batch1 = np.random.rand(32, 28, 28)
batch2 = np.random.rand(32, 28, 28)
combined_batch = np.concatenate((batch1, batch2), axis=0)
print(f"Combined Batch Shape: {combined_batch.shape}")

# %% [markdown]
# # Machine Learning Relevance
# Stacking is crucial when batching data for neural networks, augmenting datasets with new features, or ensembling predictions from multiple models.
# 
# # Common Mistakes
# - Mismatched dimensions (e.g., concatenating `(5, 4)` and `(5, 5)` along axis 0).
# - Using lists instead of tuples for array arguments in `np.concatenate((a, b))`.
# 
# # Interview Questions
# 1. What is the difference between `np.stack` and `np.concatenate`?
# 2. How do you add a bias column of 1s to a feature matrix `X`?
# 3. What does `np.vstack` do to 1D arrays vs 2D arrays?
# 4. If you have arrays A (2,3) and B (2,3), what is the shape after `np.stack((A, B), axis=-1)`?
# 5. How do you split a combined array back into its original parts?
# 
# # Practice Problems
# 1. Vertically stack two arrays `a = [10, 20]` and `b = [30, 40]`.
# 2. Add a column of ones to a `(10, 3)` random matrix.
# 3. You have 3 grayscale images of shape `(64, 64)`. Combine them into a single array of shape `(3, 64, 64)`.
# 4. Concatenate two matrices `A` `(4, 2)` and `B` `(4, 3)` horizontally.
# 5. Stack the same `A` and `B` from problem 4 vertically (Hint: You can't, why?)
# 
# # Solutions
# %%
# Solution 1
a = np.array([10, 20])
b = np.array([30, 40])
print("P1:\n", np.vstack((a, b)))

# Solution 2
mat = np.random.rand(10, 3)
ones = np.ones((10, 1))
p2 = np.hstack((mat, ones))
print("P2 shape:", p2.shape)

# Solution 3
im1, im2, im3 = np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((64, 64))
p3 = np.stack((im1, im2, im3), axis=0)
print("P3 shape:", p3.shape)

# Solution 4
A = np.zeros((4, 2))
B = np.zeros((4, 3))
p4 = np.concatenate((A, B), axis=1)
print("P4 shape:", p4.shape)

# Solution 5
# np.vstack((A, B)) would fail because dimensions along axis 1 (2 and 3) do not match.

# %% [markdown]
# # Further Reading
# - NumPy Documentation: `numpy.concatenate`, `numpy.stack`, `numpy.hstack`, `numpy.vstack`.
