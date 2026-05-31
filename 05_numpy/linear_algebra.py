# %% [markdown]
# # NumPy Linear Algebra
# 
# # Why this matters
# Machine learning algorithms, under the hood, are mostly massive linear algebra operations. Neural network forward passes are sequences of matrix multiplications. PCA requires eigenvalue decomposition.
# 
# # Learning Objectives
# - Perform dot products and matrix multiplications (`np.dot`, `@` operator, `np.matmul`).
# - Compute matrix transposes, inverses, and determinants.
# - Understand Eigenvalues and Eigenvectors (`np.linalg.eig`).
# 
# # Concept Explanation
# - **Matrix Multiplication**: Combining two matrices using the `@` operator or `np.matmul`.
# - **Dot Product**: Scalar product of vectors.
# - **Inverse**: A matrix that, when multiplied by the original, yields the identity matrix.
# - **Transpose**: Flipping a matrix over its diagonal.
# 
# # Beginner Examples
# %%
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Example 1: Dot product of vectors
dot_prod = np.dot(v1, v2)
print("Dot Product:", dot_prod)

# Example 2: Matrix Multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Using the @ operator (recommended for matrix mult)
C = A @ B
print("\nMatrix Multiplication (A @ B):\n", C)

# Example 3: Transpose
print("\nTranspose of A:\n", A.T)

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Linear Regression (Normal Equation)
# theta = (X^T * X)^-1 * X^T * y
X = np.random.rand(100, 3) # 100 samples, 3 features
y = np.random.rand(100)    # 100 targets
# Add bias column
X_b = np.c_[np.ones((100, 1)), X]
# Calculate weights
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("Calculated weights (theta):", theta)

# Example 5: Dimensionality Reduction (PCA intuition)
# Covariance matrix and Eigenvectors
features = np.random.rand(50, 4)
cov_matrix = np.cov(features, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues of covariance matrix:", eigenvalues)

# Example 6: Batch Matrix Multiplication
# Useful for attention mechanisms in Deep Learning
batch_size, seq_len, d_model = 32, 10, 64
Q = np.random.rand(batch_size, seq_len, d_model)
K = np.random.rand(batch_size, seq_len, d_model)
# Q @ K^T (Wait, K is 3D, we need to transpose the last two dimensions)
K_T = K.transpose(0, 2, 1) # Transpose sequence and feature dims
attention_scores = Q @ K_T
print("\nAttention Scores Shape:", attention_scores.shape)

# %% [markdown]
# # Machine Learning Relevance
# `np.linalg` functions are fundamental. Neural networks heavily rely on fast matrix multiplication (`@`), and classical ML models like SVMs and PCA rely on eigenvalue decompositions and matrix inversions.
# 
# # Common Mistakes
# - Confusing element-wise multiplication (`*`) with matrix multiplication (`@`).
# - Attempting to invert a singular (non-invertible) matrix.
# - Shape mismatches in matrix multiplication (inner dimensions must match).
# 
# # Interview Questions
# 1. What is the difference between `*` and `@` in NumPy?
# 2. What are the requirements for matrix `A` and `B` to be multiplied together?
# 3. How do you compute the inverse of a matrix? What if it's singular?
# 4. Explain the Normal Equation for Linear Regression and its time complexity bottleneck.
# 5. How do you compute the L2 norm of a vector in NumPy?
# 
# # Practice Problems
# 1. Compute the dot product of two vectors of size 5.
# 2. Multiply a `(3, 4)` matrix by a `(4, 2)` matrix.
# 3. Find the transpose of a `(5, 5)` matrix.
# 4. Compute the determinant of a `(2, 2)` matrix using `np.linalg.det`.
# 5. Calculate the L2 norm (magnitude) of a vector `[3, 4]` using `np.linalg.norm`.
# 
# # Solutions
# %%
# Solution 1
v1, v2 = np.random.rand(5), np.random.rand(5)
p1 = v1 @ v2

# Solution 2
A, B = np.random.rand(3, 4), np.random.rand(4, 2)
p2 = A @ B

# Solution 3
M = np.random.rand(5, 5)
p3 = M.T

# Solution 4
M2 = np.array([[1, 2], [3, 4]])
p4 = np.linalg.det(M2)
print("Determinant:", p4)

# Solution 5
v = np.array([3, 4])
p5 = np.linalg.norm(v)
print("Norm:", p5)

# %% [markdown]
# # Further Reading
# - NumPy Documentation: `numpy.linalg`.
