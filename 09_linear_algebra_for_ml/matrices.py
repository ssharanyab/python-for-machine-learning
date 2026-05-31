# %% [markdown]
# # Matrices in Linear Algebra for Machine Learning
#
# **Module**: 09 — Linear Algebra for ML
# **File**: matrices.py
# **Format**: Jupytext Light (compatible with Jupyter via `jupytext`)

# %% [markdown]
# ## Why This Matters
#
# A **matrix** is the primary data container in machine learning.
#
# | Object | Matrix Representation |
# |---|---|
# | Dataset with n samples, m features | n × m matrix |
# | Neural network layer weights | output_dim × input_dim matrix |
# | Image (grayscale) | H × W pixel matrix |
# | RGB image | H × W × 3 tensor (stack of matrices) |
# | Covariance matrix in PCA | m × m symmetric matrix |
# | Attention scores in Transformers | seq_len × seq_len matrix |
#
# Every forward pass through a neural network is a sequence of
# **matrix multiplications** followed by non-linear activations.

# %% [markdown]
# ## Learning Objectives
#
# By the end of this notebook you will be able to:
# 1. Create matrices using NumPy (zeros, ones, identity, random)
# 2. Perform element-wise and matrix-wise operations
# 3. Understand matrix shape rules and axis conventions
# 4. Slice, index, and broadcast matrices efficiently
# 5. Compute element-wise, row-wise, and column-wise aggregations
# 6. Represent real-world ML data as matrices
# 7. Understand how matrix structure connects to model architecture

# %% [markdown]
# ## Concept Explanation
#
# ### Matrix Definition
#
# An $m \times n$ matrix has $m$ rows and $n$ columns:
#
# $$A = \begin{bmatrix}
# a_{11} & a_{12} & \cdots & a_{1n} \\
# a_{21} & a_{22} & \cdots & a_{2n} \\
# \vdots & & \ddots & \vdots \\
# a_{m1} & a_{m2} & \cdots & a_{mn}
# \end{bmatrix}$$
#
# In ML convention:
# - **Rows** = data samples (observations)
# - **Columns** = features (variables)
#
# ### Special Matrices
#
# | Name | Definition | ML Use |
# |---|---|---|
# | Identity $I$ | Diagonal ones | Neutral element in products |
# | Zero matrix | All zeros | Initialisation |
# | Diagonal $D$ | Off-diagonal zeros | Feature scaling |
# | Symmetric $S$ | $S = S^T$ | Covariance, Gram matrices |
# | Orthogonal $Q$ | $Q^T Q = I$ | PCA rotations |

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#555",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

print("Libraries loaded!")

# %% [markdown]
# ## Beginner Examples
#
# ### Example 1: Creating Matrices

# %%
# --- Example 1: Creating Matrices ---
print("=" * 55)
print("EXAMPLE 1: Creating Matrices")
print("=" * 55)

# From a nested list
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"A =\n{A}\nShape: {A.shape}  dtype: {A.dtype}\n")

# Special matrices
I = np.eye(3)
print(f"Identity I (3×3):\n{I}\n")

Z = np.zeros((2, 4))
print(f"Zero matrix (2×4):\n{Z}\n")

O = np.ones((3, 3))
print(f"Ones matrix (3×3):\n{O}\n")

D = np.diag([2.0, 3.0, 5.0])
print(f"Diagonal D:\n{D}\n")

# Random matrices (used for weight init)
np.random.seed(42)
W = np.random.randn(3, 4)  # neural network weight matrix
print(f"Random weight matrix W (3×4):\n{W.round(3)}")

# %% [markdown]
# ### Example 2: Matrix Indexing and Slicing

# %%
# --- Example 2: Indexing & Slicing ---
print("=" * 55)
print("EXAMPLE 2: Indexing and Slicing")
print("=" * 55)

A = np.array([[10, 20, 30, 40],
              [50, 60, 70, 80],
              [90,100,110,120]])
print(f"A =\n{A}\nShape: {A.shape}  (3 rows × 4 cols)\n")

# Single element
print(f"A[0, 2]        = {A[0, 2]}   (row 0, col 2)")
print(f"A[-1, -1]      = {A[-1, -1]} (last row, last col)")

# Rows and columns
print(f"\nA[1, :]        = {A[1, :]}   (entire row 1)")
print(f"A[:, 2]        = {A[:, 2]}   (entire col 2)")

# Sub-matrix
print(f"\nA[0:2, 1:3] =\n{A[0:2, 1:3]}")

# Boolean masking (ML: filter samples)
mask = A > 60
print(f"\nBoolean mask (A > 60):\n{mask}")
print(f"Values > 60:  {A[mask]}")

# Fancy indexing (select specific rows — e.g., training batch)
batch_idx = [0, 2]
print(f"\nBatch rows {batch_idx}:\n{A[batch_idx, :]}")

# %% [markdown]
# ### Example 3: Element-wise Operations

# %%
# --- Example 3: Element-wise Operations ---
print("=" * 55)
print("EXAMPLE 3: Element-wise Operations")
print("=" * 55)

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

print(f"A =\n{A}\n")
print(f"B =\n{B}\n")

print(f"A + B (addition):\n{A + B}\n")
print(f"A - B (subtraction):\n{A - B}\n")
print(f"A * B (element-wise multiply — Hadamard product):\n{A * B}\n")
print(f"A / B (element-wise divide):\n{A / B}\n")
print(f"A ** 2 (element-wise square):\n{A ** 2}\n")

# Activation functions are element-wise!
print("Element-wise ReLU (activation function):")
Z = np.array([[-2.0, 3.0], [-1.0, 4.0]])
relu_Z = np.maximum(0, Z)
print(f"Z       =\n{Z}")
print(f"ReLU(Z) =\n{relu_Z}")

# %% [markdown]
# ### Example 4: Broadcasting

# %%
# --- Example 4: Broadcasting ---
print("=" * 55)
print("EXAMPLE 4: Broadcasting")
print("=" * 55)

# Broadcasting rules: shapes are compared from the right.
# Dimensions are compatible if they are equal OR one of them is 1.

A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

# Add a row vector to every row (bias addition in NN)
bias = np.array([10, 20, 30])   # shape (3,)
result = A + bias
print(f"A =\n{A}    shape: {A.shape}")
print(f"bias = {bias}  shape: {bias.shape}")
print(f"A + bias =\n{result}    ← bias added to EVERY row\n")

# Add a column vector to every column (per-sample offset)
offset = np.array([[100], [200]])   # shape (2, 1)
result2 = A + offset
print(f"offset =\n{offset}  shape: {offset.shape}")
print(f"A + offset =\n{result2}\n")

# Batch normalisation — subtract mean, divide by std (broadcasting)
data = np.array([[2.0, 4.0, 6.0],
                 [1.0, 3.0, 5.0],
                 [3.0, 5.0, 7.0]])   # shape (3,3)
col_mean = data.mean(axis=0)   # shape (3,)
col_std  = data.std(axis=0)    # shape (3,)
normalised = (data - col_mean) / (col_std + 1e-8)
print("Batch Normalisation via Broadcasting:")
print(f"data =\n{data}")
print(f"col_mean = {col_mean}")
print(f"col_std  = {col_std.round(3)}")
print(f"normalised =\n{normalised.round(4)}")

# %% [markdown]
# ### Example 5: Aggregations Along Axes

# %%
# --- Example 5: Aggregations ---
print("=" * 55)
print("EXAMPLE 5: Aggregations (axis=0 vs axis=1)")
print("=" * 55)

# Dataset: rows=samples, cols=features
# [age, income_k, years_exp]
data = np.array([[25, 50, 2],
                 [35, 80, 8],
                 [28, 60, 4],
                 [42, 95, 15],
                 [31, 70, 6]], dtype=float)

print(f"Dataset (5 samples × 3 features):\n{data}\n")
print(f"axis=0 → reduce ROWS (get a value per COLUMN = per feature)")
print(f"axis=1 → reduce COLS (get a value per ROW = per sample)\n")

print(f"Mean per feature (axis=0): {data.mean(axis=0)}")
print(f"Std  per feature (axis=0): {data.std(axis=0).round(2)}")
print(f"Max  per feature (axis=0): {data.max(axis=0)}")
print(f"\nMean per sample  (axis=1): {data.mean(axis=1).round(2)}")
print(f"Sum  per sample  (axis=1): {data.sum(axis=1)}")

# Softmax (used in classification output)
def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 3.0, 1.5]])
probs = softmax(logits)
print(f"\nLogits:\n{logits}")
print(f"Softmax probs:\n{probs.round(4)}")
print(f"Row sums (should be 1.0): {probs.sum(axis=1)}")

# %% [markdown]
# ## Intermediate Examples
#
# ### Example 6: Covariance Matrix

# %%
# --- Example 6: Covariance Matrix (PCA foundation) ---
print("=" * 55)
print("EXAMPLE 6: Covariance Matrix")
print("=" * 55)

np.random.seed(7)
# Correlated data: height and weight
n = 200
height = np.random.normal(170, 10, n)
weight = 0.6 * height + np.random.normal(0, 5, n)  # correlated with height

X = np.column_stack([height, weight])  # shape (200, 2)
X_centered = X - X.mean(axis=0)

# Manual covariance matrix: C = (X^T X) / (n-1)
C_manual = (X_centered.T @ X_centered) / (n - 1)
# NumPy covariance
C_np = np.cov(X.T)

print(f"Data shape: {X.shape}")
print(f"\nCovariance matrix (manual):\n{C_manual.round(3)}")
print(f"\nCovariance matrix (np.cov):\n{C_np.round(3)}")
print("\nInterpretation:")
print(f"  Var(height)       = {C_np[0,0]:.3f}")
print(f"  Var(weight)       = {C_np[1,1]:.3f}")
print(f"  Cov(h, w)         = {C_np[0,1]:.3f}  (positive → correlated)")
print(f"  Correlation coef  = {C_np[0,1]/np.sqrt(C_np[0,0]*C_np[1,1]):.4f}")

# %% [markdown]
# ### Example 7: Matrix as a Linear Transformation

# %%
# --- Example 7: Matrix as Linear Transformation ---
print("=" * 55)
print("EXAMPLE 7: Matrices as Linear Transformations")
print("=" * 55)

# A matrix transforms vectors from one space to another.
# Every neural network layer IS a linear transformation.

# Create a unit circle
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.stack([np.cos(theta), np.sin(theta)])  # shape (2, 100)

# Transformation matrices
transforms = {
    "Scaling (2x, 0.5y)": np.array([[2.0, 0.0], [0.0, 0.5]]),
    "Rotation 45°":        np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                      [np.sin(np.pi/4),  np.cos(np.pi/4)]]),
    "Shear":               np.array([[1.0, 0.5], [0.0, 1.0]]),
    "Reflection (x-axis)": np.array([[1.0, 0.0], [0.0, -1.0]]),
}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Matrices as Linear Transformations of the Unit Circle",
             fontsize=14, color="white", fontweight="bold")

for ax, (name, T) in zip(axes, transforms.items()):
    transformed = T @ circle   # (2×2) @ (2×100) = (2×100)
    ax.plot(circle[0], circle[1], "--", color="#555", lw=1.5, label="Original")
    ax.fill(transformed[0], transformed[1], alpha=0.3, color="#00d4ff")
    ax.plot(transformed[0], transformed[1], color="#00d4ff", lw=2, label="Transformed")
    ax.set_title(name, color="white", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#444", lw=1)
    ax.axvline(0, color="#444", lw=1)
    print(f"\nTransformation matrix '{name}':\n{T}")

plt.tight_layout()
plt.savefig("linear_transformations.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ### Example 8: Stacking and Reshaping (CNN preprocessing)

# %%
# --- Example 8: Reshaping for Deep Learning ---
print("=" * 55)
print("EXAMPLE 8: Reshaping — Images for CNNs")
print("=" * 55)

np.random.seed(0)
# Simulate a batch of 4 grayscale images, 8×8 pixels
batch_size = 4
H, W = 8, 8
images = np.random.randint(0, 256, size=(batch_size, H, W), dtype=np.uint8)
print(f"Batch of images shape: {images.shape}  (batch × H × W)")

# Flatten each image to a vector (for fully connected layers)
flat = images.reshape(batch_size, -1)   # shape (4, 64)
print(f"Flattened for FC layer: {flat.shape}")

# Batch normalise flattened batch
flat_f = flat.astype(float)
normalised = (flat_f - flat_f.mean(axis=1, keepdims=True)) / \
             (flat_f.std(axis=1, keepdims=True) + 1e-8)
print(f"Normalised batch shape: {normalised.shape}")
print(f"Row means (should ≈ 0): {normalised.mean(axis=1).round(6)}")

# Add channel dimension for CNN input: (batch, channels, H, W)
images_rgb = np.random.randint(0, 256, size=(2, 32, 32, 3))   # HWC format
images_chw = images_rgb.transpose(0, 3, 1, 2)                  # CHW format (PyTorch)
print(f"\nImage batch HWC: {images_rgb.shape}  → CHW: {images_chw.shape}")

# %% [markdown]
# ## Machine Learning Relevance
#
# ### Example 9: Dataset as a Design Matrix

# %%
# --- Example 9: Design Matrix (X) in Linear Regression ---
print("=" * 55)
print("EXAMPLE 9: Design Matrix for Linear Regression")
print("=" * 55)

# The design matrix X stores ALL training samples.
# Shape: (n_samples, n_features) — the universal ML data format.

np.random.seed(42)
n, d = 100, 3  # 100 samples, 3 features
feature_names = ["sqft", "bedrooms", "age"]

X = np.column_stack([
    np.random.normal(1500, 300, n),   # sqft
    np.random.randint(1, 6, n).astype(float),  # bedrooms
    np.random.randint(1, 30, n).astype(float),  # age
])
w_true = np.array([0.5, 20.0, -2.0])
b_true = 100.0
y = X @ w_true + b_true + np.random.randn(n) * 10

print(f"Design matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")
print(f"\nFirst 5 rows of X:\n{X[:5].round(1)}")

# Closed-form solution: w = (X^T X)^{-1} X^T y  (Normal Equation)
X_b = np.column_stack([np.ones(n), X])   # add bias column
w_hat = np.linalg.lstsq(X_b, y, rcond=None)[0]
print(f"\nNormal Equation solution:")
print(f"  Estimated bias:    {w_hat[0]:.4f}  (true: {b_true})")
for i, (name, wt, we) in enumerate(zip(feature_names, w_true, w_hat[1:])):
    print(f"  w_{name:<10}: estimated={we:.4f}  true={wt}")

# %% [markdown]
# ### Example 10: Visualizing Matrix Operations as Heatmaps

# %%
# --- Example 10: Matrix Heatmap Visualisation ---
print("=" * 55)
print("EXAMPLE 10: Heatmap Visualisation of Matrices")
print("=" * 55)

np.random.seed(5)
n_features = 6
feature_labels = [f"F{i}" for i in range(n_features)]

# Simulated covariance matrix
A = np.random.randn(30, n_features)
cov = np.cov(A.T)
corr = np.corrcoef(A.T)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Matrix Heatmaps in ML", fontsize=14, color="white", fontweight="bold")

cmap = plt.cm.RdBu_r

matrices = [cov, corr, np.abs(corr)]
titles   = ["Covariance Matrix", "Correlation Matrix", "|Correlation| (Feature Importance)"]

for ax, mat, title in zip(axes, matrices, titles):
    im = ax.imshow(mat, cmap=cmap, vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xticks(range(n_features)); ax.set_xticklabels(feature_labels)
    ax.set_yticks(range(n_features)); ax.set_yticklabels(feature_labels)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(n_features):
        for j in range(n_features):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(mat[i,j]) < 0.5 else "white")

plt.tight_layout()
plt.savefig("matrix_heatmaps.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ## Common Mistakes

# %%
print("COMMON MISTAKE 1: `*` vs `@` for matrix operations")
print("-" * 50)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A * B (Hadamard/element-wise):\n{A * B}")
print(f"A @ B (matrix multiplication):\n{A @ B}")

print("\nCOMMON MISTAKE 2: Shape (n,) vs shape (n,1)")
print("-" * 50)
v = np.array([1, 2, 3])
print(f"v.shape = {v.shape}   — 1D array, NOT a column vector")
print(f"v.reshape(-1,1).shape = {v.reshape(-1,1).shape}  — proper column vector")

print("\nCOMMON MISTAKE 3: Axis confusion in aggregations")
print("-" * 50)
M = np.array([[1, 2, 3], [4, 5, 6]])
print(f"M =\n{M}")
print(f"M.sum(axis=0) = {M.sum(axis=0)}  ← sum DOWN columns (per-feature total)")
print(f"M.sum(axis=1) = {M.sum(axis=1)}  ← sum ACROSS rows (per-sample total)")

print("\nCOMMON MISTAKE 4: Shallow copy vs deep copy")
print("-" * 50)
A = np.array([[1, 2], [3, 4]])
B = A        # view (same memory)
C = A.copy() # deep copy
B[0, 0] = 999
print(f"After B[0,0]=999: A[0,0]={A[0,0]} (changed!), C[0,0]={C[0,0]} (safe)")

# %% [markdown]
# ## Interview Questions
#
# **Q1. How is a dataset represented as a matrix in ML, and what do rows and columns mean?**
#
# **A:** A dataset is stored as a **design matrix** $X \in \mathbb{R}^{n \times m}$,
# where each of the $n$ rows is a data sample and each of the $m$ columns is a feature.
# This convention allows efficient batch operations: computing predictions for all
# samples at once via $\hat{y} = X\mathbf{w} + b$ (a single matrix-vector multiply).
#
# **Q2. What is the difference between element-wise multiplication and matrix multiplication?**
#
# **A:** Element-wise (Hadamard) multiplication $A \odot B$ multiplies corresponding
# elements; both matrices must have the same shape. Matrix multiplication $AB$
# computes dot products of rows and columns; $A$ must be $m\times k$ and $B$ must be
# $k\times n$ — the inner dimensions must match. In ML, Hadamard products appear
# in gradient computations (LSTM gates), while matrix multiplication implements
# linear layer forward passes.
#
# **Q3. What is broadcasting and why is it important in NumPy for ML?**
#
# **A:** Broadcasting automatically expands dimensions of size 1 to match the other
# operand's shape, enabling vectorised operations without explicit loops.
# Examples: subtracting column means from a data matrix (`X - X.mean(axis=0)`),
# adding bias vectors to batches of activations. It is essential for performance —
# Python loops over large arrays are orders of magnitude slower.
#
# **Q4. What is a covariance matrix and how is it used in ML?**
#
# **A:** The covariance matrix $\Sigma \in \mathbb{R}^{m\times m}$ captures pairwise
# linear relationships between all $m$ features. $\Sigma_{ij}$ is positive if
# features $i$ and $j$ tend to increase together. PCA decomposes $\Sigma$ into
# eigenvalues and eigenvectors to find the directions of maximum variance,
# enabling dimensionality reduction.
#
# **Q5. Explain why neural network weight matrices have shape (output_dim, input_dim).**
#
# **A:** For a layer with input $\mathbf{x} \in \mathbb{R}^{d_{in}}$ and weights
# $W \in \mathbb{R}^{d_{out} \times d_{in}}$, the output is
# $\mathbf{z} = W\mathbf{x} + \mathbf{b} \in \mathbb{R}^{d_{out}}$.
# The matrix-vector product $W\mathbf{x}$ has shape $(d_{out},)$, which is the
# activation vector of the next layer. With batches: $Z = XW^T$ where
# $X \in \mathbb{R}^{n \times d_{in}}$ gives $Z \in \mathbb{R}^{n \times d_{out}}$.

# %% [markdown]
# ## Practice Problems
#
# 1. **Matrix Arithmetic**: Create a 4×4 matrix of your choice. Compute its row-means,
#    column-means, and the overall mean. Subtract the overall mean from every element.
#
# 2. **Broadcasting**: Given a 5×3 feature matrix `X` and a 1D mean vector `mu` of
#    shape (3,), normalise `X` by subtracting `mu` and dividing by the std per feature.
#
# 3. **Correlation Matrix**: Generate a 100×4 random dataset. Compute its correlation
#    matrix using `np.corrcoef`. Which two features are most correlated?
#
# 4. **Linear Transformation**: Apply the rotation matrix for 30° to a set of 50 random
#    2D points. Plot the original and transformed point clouds.
#
# 5. **Batch Softmax**: Implement a batched softmax function that accepts a matrix
#    `Z` of shape (n, k) and returns probabilities of the same shape.
#    Verify rows sum to 1.

# %% [markdown]
# ## Solutions

# %%
# Solution 1
print("SOLUTION 1")
M = np.arange(1, 17, dtype=float).reshape(4, 4)
row_means  = M.mean(axis=1, keepdims=True)
col_means  = M.mean(axis=0, keepdims=True)
overall    = M.mean()
M_centred  = M - overall
print(f"M:\n{M}")
print(f"Row means:    {M.mean(axis=1).round(2)}")
print(f"Col means:    {M.mean(axis=0).round(2)}")
print(f"Overall mean: {overall}")
print(f"M - mean:\n{M_centred}")

# Solution 2
print("\nSOLUTION 2: Normalisation via Broadcasting")
np.random.seed(1)
X = np.random.randn(5, 3) * np.array([10, 2, 100]) + np.array([50, 5, 1000])
mu  = X.mean(axis=0)
sig = X.std(axis=0)
X_norm = (X - mu) / (sig + 1e-8)
print(f"Original X:\n{X.round(2)}")
print(f"Normalised (should have mean≈0, std≈1):")
print(f"  mean = {X_norm.mean(axis=0).round(6)}")
print(f"  std  = {X_norm.std(axis=0).round(6)}")

# Solution 3
print("\nSOLUTION 3: Correlation Matrix")
np.random.seed(3)
data = np.random.randn(100, 4)
data[:, 2] = data[:, 0] * 2 + np.random.randn(100) * 0.1  # make 0 and 2 correlated
corr = np.corrcoef(data.T)
print(f"Correlation matrix:\n{corr.round(3)}")
# Find max off-diagonal
np.fill_diagonal(corr, 0)
idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
print(f"Most correlated pair: features {idx[0]} and {idx[1]}")

# Solution 4
print("\nSOLUTION 4: Rotation Transformation")
angle = np.radians(30)
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])
pts = np.random.randn(50, 2)
pts_rot = (R @ pts.T).T
print(f"Rotation matrix R (30°):\n{R.round(3)}")
print(f"First point before: {pts[0].round(3)}")
print(f"First point after:  {pts_rot[0].round(3)}")

# Solution 5
print("\nSOLUTION 5: Batch Softmax")
def batch_softmax(Z: np.ndarray) -> np.ndarray:
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

Z_test = np.array([[1.0, 2.0, 3.0],
                   [0.1, 0.5, 0.4]])
probs = batch_softmax(Z_test)
print(f"Input logits:\n{Z_test}")
print(f"Softmax probs:\n{probs.round(4)}")
print(f"Row sums: {probs.sum(axis=1)}")

# %% [markdown]
# ## Further Reading
#
# - **3Blue1Brown — Essence of Linear Algebra** (YouTube):
#   <https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab>
# - **NumPy Broadcasting Guide**:
#   <https://numpy.org/doc/stable/user/basics.broadcasting.html>
# - **CS231n — Linear Algebra Review** (Stanford):
#   <https://cs231n.github.io/python-numpy-tutorial/>
# - **Mathematics for Machine Learning (Deisenroth, Faisal, Ong)**:
#   <https://mml-book.github.io/>
# - **Fast.ai — Matrix Multiplication from scratch**:
#   <https://www.fast.ai/>
