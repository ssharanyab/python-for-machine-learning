# %% [markdown]
# # Array Broadcasting: NumPy's Superpower
# 
# # Why this matters
# Broadcasting is arguably the most powerful and unique feature of NumPy. It allows mathematical operations between arrays of different shapes without manually creating `for` loops or explicitly duplicating data to make their shapes match. When you add a bias term to the output of a neural network layer (`output + bias`), broadcasting is what allows a 1D bias vector to be correctly added to a 2D batch of outputs automatically and instantly.
# 
# # Learning Objectives
# 1. Understand the rules of broadcasting.
# 2. Perform arithmetic operations between scalars, 1D vectors, and 2D matrices.
# 3. Diagnose and fix common broadcasting shape mismatches.
# 
# # Concept Explanation
# When operating on two arrays, NumPy compares their shapes element-wise, starting from the trailing dimensions (rightmost) and working leftwards.
# Two dimensions are compatible if:
# 1. They are equal, or
# 2. One of them is 1.
# 
# If one array has fewer dimensions than the other, NumPy implicitly pads its shape with ones on the left.
# Once shapes are compatible, NumPy "broadcasts" (virtually stretches) the array with dimension 1 along that axis to match the other array, performing the calculation element-wise.
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: Scalar and 1D Array
# Adding a scalar to an array
# scalar '5' (shape: ()) is broadcasted to [5, 5, 5] (shape: (3,))
arr = np.array([1, 2, 3])
print("Array + 5:", arr + 5)

# Example 2: 1D Array and 2D Matrix
# Centering data by subtracting the mean of each column
# Matrix shape: (3, 2). Mean vector shape: (2,)
data = np.array([
    [10, 20],
    [30, 40],
    [50, 60]
])
mean = np.array([30, 40]) # Shape (2,) 

# Broadcasting rules check:
# data: (3, 2)
# mean:    (2,) -> Padded to (1, 2) -> Broadcasted to (3, 2)
centered_data = data - mean
print("\nCentered Data:\n", centered_data)

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 3: Adding a Column Vector to a Matrix
# Suppose we want to add a different value to each row
# Matrix shape: (3, 2). Vector shape: (3,)
row_vector = np.array([100, 200, 300]) 

# Try: data + row_vector
# data:        (3, 2)
# row_vector:     (3,) -> Padded to (1, 3).
# Mismatch! 2 != 3. This will throw an error!

# Fix: Reshape row_vector to be a column vector (3, 1)
col_vector = row_vector.reshape(3, 1)
# data:       (3, 2)
# col_vector: (3, 1) -> Broadcasted to (3, 2)
result = data + col_vector
print("Matrix + Column Vector:\n", result)

# Example 4: Outer Product via Broadcasting
# A column vector (3, 1) multiplied by a row vector (1, 4)
a = np.array([1, 2, 3]).reshape(3, 1)
b = np.array([10, 20, 30, 40]).reshape(1, 4)
# a: (3, 1) -> Broadcasted to (3, 4)
# b: (1, 4) -> Broadcasted to (3, 4)
# Result: (3, 4)
outer_product = a * b
print("\nOuter Product via Broadcasting:\n", outer_product)

# %% [markdown]
# # Machine Learning Relevance
# **Feature Scaling (Standardization):**
# To scale features `X` (shape `(samples, features)`) to have 0 mean and unit variance, you calculate `mean` and `std` across `axis=0` (resulting in shape `(features,)`). Broadcasting allows you to simply write:
# `X_scaled = (X - mean) / std`
# 
# **Image Operations:**
# If you have an image `(height, width, 3)` and want to tint the RGB channels by multiplying them by `[0.5, 1.0, 0.5]`, broadcasting aligns `(height, width, 3)` with `(3,)` seamlessly.
# 
# # Common Mistakes
# 1. **Assuming `(N,)` acts as a column vector:** A 1D array of shape `(N,)` aligns with the LAST dimension of a matrix (i.e., it acts across columns). If you want it to align with rows, you MUST reshape it to `(N, 1)`.
# 2. **Mismatched Trailing Dimensions:** Trying to add an array of shape `(256, 256, 3)` and `(256,)` will fail. The trailing dimensions are 3 and 256, which are incompatible.
# 
# # Interview Questions
# 1. **Explain how NumPy's broadcasting rules would evaluate adding an array of shape `(100, 50, 3)` to an array of shape `(50, 1)`.**
#    *Answer:* 
#    - Shapes: `(100, 50, 3)` and `(50, 1)`
#    - Pad shorter: `(100, 50, 3)` and `(1, 50, 1)`
#    - Compare right-to-left:
#      - 3 and 1: Compatible (1 becomes 3)
#      - 50 and 50: Compatible
#      - 100 and 1: Compatible (1 becomes 100)
#    - Result shape: `(100, 50, 3)`.
# 2. **Why is broadcasting faster than a Python `for` loop?**
#    *Answer:* Broadcasting occurs in highly optimized C code under the hood. It also avoids making physical copies of the broadcasted data in memory, making it highly memory efficient.
# 3. **If `A` is `(4, 3)` and `B` is `(4,)`, will `A + B` work? If not, how do you fix it?**
#    *Answer:* It will fail because trailing dimensions (3 and 4) don't match. Fix it by making B a column vector: `A + B.reshape(4, 1)` or `A + B[:, np.newaxis]`.
# 4. **How do you subtract the row-wise mean from every row in a matrix `X` of shape `(M, N)`?**
#    *Answer:* `X - X.mean(axis=1, keepdims=True)`. The `keepdims` ensures the mean vector has shape `(M, 1)` allowing it to broadcast across `(M, N)`.
# 5. **What is the result shape of `np.ones((5, 1, 4)) + np.ones((3, 4))`?**
#    *Answer:* Padded B: `(1, 3, 4)`. Broadcasting: `(5, 1, 4)` and `(1, 3, 4)` -> `(5, 3, 4)`.
# 
# # Practice Problems
# 1. Add the scalar 100 to a 2D matrix of shape `(3, 3)`.
# 2. Create a matrix of shape `(4, 5)`. Subtract the array `[1, 2, 3, 4, 5]` from every row using broadcasting.
# 3. Create a matrix of shape `(4, 5)`. Subtract the array `[10, 20, 30, 40]` from every column. (Hint: Reshape).
# 4. Multiply a column vector of shape `(5, 1)` with a row vector of shape `(1, 5)`. What is the result shape?
# 5. Given a 3D image tensor of shape `(28, 28, 3)`, multiply the 3 color channels by the vector `[0.299, 0.587, 0.114]` (Luma transform coefficients).
# 
# # Solutions
# 
# %%
# Solution 1
mat1 = np.zeros((3, 3))
print("Q1:\n", mat1 + 100)

# Solution 2
mat2 = np.zeros((4, 5))
vec2 = np.array([1, 2, 3, 4, 5])
print("Q2:\n", mat2 - vec2)

# Solution 3
mat3 = np.zeros((4, 5))
vec3 = np.array([10, 20, 30, 40]).reshape(-1, 1)
print("Q3:\n", mat3 - vec3)

# Solution 4
col = np.arange(5).reshape(5, 1)
row = np.arange(5).reshape(1, 5)
print("Q4 shape:", (col * row).shape)

# Solution 5
img = np.ones((28, 28, 3))
coeffs = np.array([0.299, 0.587, 0.114])
luma = img * coeffs
print("Q5 shape:", luma.shape)

# %% [markdown]
# # Further Reading
# - NumPy Broadcasting Guide: https://numpy.org/doc/stable/user/basics.broadcasting.html
