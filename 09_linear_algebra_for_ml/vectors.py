# %% [markdown]
# # Vectors in Linear Algebra for Machine Learning
#
# **Module**: 09 — Linear Algebra for ML
# **File**: vectors.py
# **Format**: Jupytext Light (compatible with Jupyter via `jupytext`)

# %% [markdown]
# ## Why This Matters
#
# Vectors are the **fundamental data structure of machine learning**.
# Every data point, every weight in a neural network, every word embedding,
# and every gradient in backpropagation is represented as a vector.
#
# Understanding vectors is not optional — it is the bedrock on which
# all of ML mathematics is built.
#
# | Use Case | Vector Role |
# |---|---|
# | A single data sample (e.g., house features) | Feature vector |
# | Word2Vec / GloVe word embeddings | High-dimensional vectors |
# | Neural network weights per neuron | Weight vector |
# | Gradient descent update | Gradient vector |
# | Cosine similarity in recommendation systems | Direction comparison |

# %% [markdown]
# ## Learning Objectives
#
# By the end of this notebook you will be able to:
# 1. Create and manipulate vectors using NumPy
# 2. Compute vector addition, subtraction, and scalar multiplication
# 3. Calculate vector norms (L1, L2, L∞)
# 4. Understand and compute dot products and their geometric meaning
# 5. Visualize 2D/3D vectors with Matplotlib
# 6. Explain cosine similarity and use it for text similarity
# 7. Connect every vector operation to a real ML application

# %% [markdown]
# ## Concept Explanation
#
# ### What is a Vector?
#
# A **vector** is an ordered list of numbers. Geometrically, it represents
# a direction and magnitude in n-dimensional space.
#
# $$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$
#
# ### Core Operations
#
# | Operation | Formula | ML Use |
# |---|---|---|
# | Addition | $\mathbf{a} + \mathbf{b}$ | Combining gradients |
# | Scalar mult | $\alpha \mathbf{v}$ | Learning rate scaling |
# | Dot product | $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$ | Similarity, projections |
# | L2 Norm | $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ | Regularization |
# | Cosine sim | $\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$ | Text/rec similarity |

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

# Consistent style
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#444",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333",
    "grid.alpha": 0.5,
})

print("Libraries loaded successfully!")

# %% [markdown]
# ## Beginner Examples
#
# ### Example 1: Creating Vectors

# %%
# --- Example 1: Creating Vectors ---
print("=" * 55)
print("EXAMPLE 1: Creating Vectors")
print("=" * 55)

# 1D vector (row vector)
v_row = np.array([1, 2, 3])
print(f"Row vector:    {v_row}   shape: {v_row.shape}")

# Column vector (2D array with shape (n,1))
v_col = np.array([[1], [2], [3]])
print(f"Column vector:\n{v_col}   shape: {v_col.shape}")

# Zero vector
v_zero = np.zeros(5)
print(f"Zero vector:   {v_zero}")

# Ones vector
v_ones = np.ones(4)
print(f"Ones vector:   {v_ones}")

# Range vector
v_range = np.arange(0, 10, 2)
print(f"Range vector:  {v_range}")

# Random vector (ML: random weight initialisation)
np.random.seed(42)
v_rand = np.random.randn(5)
print(f"Random vector: {v_rand.round(3)}")

# %% [markdown]
# ### Example 2: Vector Arithmetic

# %%
# --- Example 2: Vector Arithmetic ---
print("=" * 55)
print("EXAMPLE 2: Vector Arithmetic")
print("=" * 55)

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print()

# Addition (gradient accumulation in ML)
print(f"a + b          = {a + b}")

# Subtraction (residuals in regression: y_true - y_pred)
print(f"b - a          = {b - a}")

# Scalar multiplication (learning rate × gradient)
lr = 0.01
gradient = np.array([0.5, -1.2, 0.8])
update = lr * gradient
print(f"\nLearning rate = {lr}")
print(f"Gradient      = {gradient}")
print(f"Weight update = lr * gradient = {update}")

# Element-wise multiplication
print(f"\na * b (elem-wise) = {a * b}")

# %% [markdown]
# ### Example 3: Vector Norms

# %%
# --- Example 3: Vector Norms ---
print("=" * 55)
print("EXAMPLE 3: Vector Norms (Magnitude / Length)")
print("=" * 55)

v = np.array([3.0, 4.0])

# L1 norm (Manhattan distance) — used in L1 regularisation (Lasso)
l1 = np.linalg.norm(v, ord=1)
print(f"v = {v}")
print(f"L1 norm  (|3| + |4|)         = {l1:.4f}")

# L2 norm (Euclidean distance) — used in L2 regularisation (Ridge)
l2 = np.linalg.norm(v, ord=2)
print(f"L2 norm  (sqrt(9+16))        = {l2:.4f}   ← the '5' in a 3-4-5 triangle")

# L-infinity norm (max abs value) — used in adversarial ML
linf = np.linalg.norm(v, ord=np.inf)
print(f"L∞ norm  (max(|3|, |4|))     = {linf:.4f}")

# Unit vector (normalisation — crucial in cosine similarity)
unit = v / l2
print(f"\nUnit vector of v             = {unit}")
print(f"Verify ‖unit‖₂               = {np.linalg.norm(unit):.6f}  (should be 1.0)")

# %% [markdown]
# ### Example 4: Dot Product

# %%
# --- Example 4: Dot Product ---
print("=" * 55)
print("EXAMPLE 4: Dot Product")
print("=" * 55)

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Method 1: np.dot
dot_np = np.dot(a, b)
# Method 2: @ operator
dot_at = a @ b
# Method 3: Manual
dot_manual = sum(a[i] * b[i] for i in range(len(a)))

print(f"a = {a}")
print(f"b = {b}")
print(f"\nnp.dot(a, b)      = {dot_np}")
print(f"a @ b             = {dot_at}")
print(f"Manual sum        = {dot_manual}")

# Geometric interpretation
theta_rad = np.arccos(dot_np / (np.linalg.norm(a) * np.linalg.norm(b)))
theta_deg = np.degrees(theta_rad)
print(f"\nAngle between a and b = {theta_deg:.2f}°")

# Perpendicular vectors have dot product = 0
x_axis = np.array([1, 0])
y_axis = np.array([0, 1])
print(f"\nDot product of perpendicular vectors: {np.dot(x_axis, y_axis)}")

# %% [markdown]
# ### Example 5: Visualizing Vectors

# %%
# --- Example 5: Visualizing 2D Vectors ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Vector Operations — Visualized", fontsize=16, fontweight="bold", color="white")

# --- Left: Basic vectors ---
ax1 = axes[0]
ax1.set_title("Vector Addition & Subtraction", color="white", fontsize=13)
origin = np.array([0, 0])

a2d = np.array([3, 1])
b2d = np.array([1, 3])
ab_sum = a2d + b2d

colors = ["#00d4ff", "#ff6b6b", "#98fb98"]
labels = [f"a = {a2d}", f"b = {b2d}", f"a+b = {ab_sum}"]
vectors = [a2d, b2d, ab_sum]

for vec, color, label in zip(vectors, colors, labels):
    ax1.annotate("", xy=vec, xytext=origin,
                 arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
    ax1.text(vec[0] * 0.55, vec[1] * 0.55 + 0.1, label, color=color, fontsize=11)

# Dashed parallelogram
ax1.plot([a2d[0], ab_sum[0]], [a2d[1], ab_sum[1]], "--", color="#ff6b6b", alpha=0.5, lw=1)
ax1.plot([b2d[0], ab_sum[0]], [b2d[1], ab_sum[1]], "--", color="#00d4ff", alpha=0.5, lw=1)

ax1.set_xlim(-0.5, 5)
ax1.set_ylim(-0.5, 5)
ax1.axhline(0, color="#444", lw=1)
ax1.axvline(0, color="#444", lw=1)
ax1.grid(True, alpha=0.3)
ax1.set_aspect("equal")

# --- Right: Cosine similarity ---
ax2 = axes[1]
ax2.set_title("Cosine Similarity — Angle Between Vectors", color="white", fontsize=13)

v1 = np.array([4, 1])
v2 = np.array([1, 4])
v3 = np.array([-4, 1])

for vec, color, label in zip([v1, v2, v3],
                               ["#00d4ff", "#ffd700", "#ff6b6b"],
                               ["v1", "v2", "v3 (opposite dir)"]):
    ax2.annotate("", xy=vec, xytext=origin,
                 arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
    offset = 0.2
    ax2.text(vec[0] + offset, vec[1] + offset, label, color=color, fontsize=11)

    cos_v1 = np.dot(v1, vec) / (np.linalg.norm(v1) * np.linalg.norm(vec))
    print(f"  cos_sim(v1, {label}) = {cos_v1:.4f}")

ax2.set_xlim(-5, 5)
ax2.set_ylim(-1, 5)
ax2.axhline(0, color="#444", lw=1)
ax2.axvline(0, color="#444", lw=1)
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal")

plt.tight_layout()
plt.savefig("vector_visualization.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Figure saved as 'vector_visualization.png'")

# %% [markdown]
# ## Intermediate Examples
#
# ### Example 6: Cosine Similarity for Text (NLP)

# %%
# --- Example 6: Cosine Similarity in NLP ---
print("=" * 55)
print("EXAMPLE 6: Cosine Similarity — Text Embeddings")
print("=" * 55)

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two non-zero vectors."""
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

# Simulated word embeddings (in practice: Word2Vec / GloVe outputs)
embeddings = {
    "king":   np.array([0.9, 0.1, 0.8, 0.3]),
    "queen":  np.array([0.8, 0.9, 0.7, 0.3]),
    "man":    np.array([0.7, 0.1, 0.2, 0.1]),
    "woman":  np.array([0.6, 0.9, 0.2, 0.1]),
    "apple":  np.array([0.1, 0.0, 0.0, 0.9]),
}

print("Cosine Similarity Matrix:")
words = list(embeddings.keys())
for w1 in words:
    for w2 in words:
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        print(f"  sim({w1:<6}, {w2:<6}) = {sim:.4f}")
    print()

# Classic king - man + woman ≈ queen analogy
analogy = embeddings["king"] - embeddings["man"] + embeddings["woman"]
print("king - man + woman =", analogy.round(3))
print(f"  → Similarity to 'queen': {cosine_similarity(analogy, embeddings['queen']):.4f}")
print(f"  → Similarity to 'apple': {cosine_similarity(analogy, embeddings['apple']):.4f}")

# %% [markdown]
# ### Example 7: Projection and Basis Change

# %%
# --- Example 7: Vector Projection ---
print("=" * 55)
print("EXAMPLE 7: Vector Projection (used in PCA, Gram-Schmidt)")
print("=" * 55)

def project(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Project vector v onto unit vector u."""
    u_unit = u / np.linalg.norm(u)
    scalar = np.dot(v, u_unit)
    return scalar * u_unit

a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])  # x-axis

proj = project(a, b)
print(f"a            = {a}")
print(f"b (x-axis)   = {b}")
print(f"proj_b(a)    = {proj}   ← a's shadow on x-axis")

# Projection onto arbitrary direction (PCA principal component)
pc1 = np.array([1.0, 1.0])  # principal component direction
proj_pc1 = project(a, pc1)
print(f"\nProjection onto PC1 direction {pc1}:")
print(f"proj_pc1(a)  = {proj_pc1.round(4)}")

scalar_proj = np.dot(a, pc1 / np.linalg.norm(pc1))
print(f"Scalar projection (coordinate in PC1 space) = {scalar_proj:.4f}")

# %% [markdown]
# ### Example 8: Feature Vectors in ML

# %%
# --- Example 8: Feature Vectors in a Real Dataset ---
print("=" * 55)
print("EXAMPLE 8: Feature Vectors — Real ML Context")
print("=" * 55)

# House price dataset — each sample is a feature vector
feature_names = ["bedrooms", "bathrooms", "sqft", "age_years", "garage"]
house1 = np.array([3, 2, 1500, 10, 1], dtype=float)
house2 = np.array([4, 3, 2200, 5,  2], dtype=float)

print("House 1 feature vector:", house1)
print("House 2 feature vector:", house2)

# Euclidean distance (used in KNN)
distance = np.linalg.norm(house1 - house2)
print(f"\nEuclidean distance between houses: {distance:.2f}")

# Normalise feature vectors (z-score per feature across the dataset)
data = np.array([house1, house2,
                 [2, 1, 900, 25, 0],
                 [5, 4, 3000, 2, 2]], dtype=float)

mean = data.mean(axis=0)
std  = data.std(axis=0)
data_normalised = (data - mean) / (std + 1e-8)

print("\nOriginal data matrix (rows=samples, cols=features):")
print(data)
print("\nNormalised feature vectors:")
print(data_normalised.round(3))
print(f"\nFeature means: {mean.round(1)}")
print(f"Feature stds:  {std.round(1)}")

# %% [markdown]
# ## Machine Learning Relevance
#
# ### Example 9: Gradient Descent Step as Vector Update

# %%
# --- Example 9: Gradient Descent — Pure Vector Math ---
print("=" * 55)
print("EXAMPLE 9: Gradient Descent via Vector Operations")
print("=" * 55)

np.random.seed(0)

# Simple linear regression: y = w·x + b, minimise MSE
# True weights
w_true = np.array([2.5, -1.0, 0.7])
b_true = 3.0
n_samples = 50

# Generate data
X = np.random.randn(n_samples, 3)
y = X @ w_true + b_true + np.random.randn(n_samples) * 0.2

# Initialise weights randomly
w = np.zeros(3)
b = 0.0
lr = 0.05
history = []

for epoch in range(100):
    # Forward pass: predictions = X @ w + b   (matrix-vector product)
    y_pred = X @ w + b

    # Loss (MSE)
    loss = np.mean((y_pred - y) ** 2)
    history.append(loss)

    # Gradients — each is a VECTOR
    grad_w = (2 / n_samples) * (X.T @ (y_pred - y))   # shape (3,)
    grad_b = (2 / n_samples) * np.sum(y_pred - y)      # scalar

    # Vector update
    w -= lr * grad_w
    b -= lr * grad_b

print(f"True weights:      {w_true}")
print(f"Learned weights:   {w.round(4)}")
print(f"True bias:         {b_true}")
print(f"Learned bias:      {b:.4f}")
print(f"\nInitial loss:      {history[0]:.4f}")
print(f"Final loss:        {history[-1]:.4f}")

# Plot convergence
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history, color="#00d4ff", lw=2)
ax.set_title("Gradient Descent — Loss Convergence", color="white", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gradient_descent_convergence.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ### Example 10: Cosine Similarity in a Recommendation System

# %%
# --- Example 10: Movie Recommendation via Cosine Similarity ---
print("=" * 55)
print("EXAMPLE 10: Recommendation System — Cosine Similarity")
print("=" * 55)

# User-item rating matrix (rows=users, cols=movies)
# 0 means not rated
movies = ["Inception", "Interstellar", "The Dark Knight", "Titanic", "Notebook"]
ratings = np.array([
    [5, 4, 5, 1, 1],   # User A (action fan)
    [4, 5, 4, 0, 2],   # User B (sci-fi fan)
    [1, 2, 1, 5, 5],   # User C (romance fan)
    [0, 4, 5, 1, 0],   # User D (unknown)
])
user_names = ["Alice", "Bob", "Carol", "Dave (target)"]

# Find most similar users to Dave using cosine similarity
dave = ratings[3]
print(f"Dave's ratings: {dave}  (0 = not seen)\n")

for i, (name, user_vec) in enumerate(zip(user_names[:3], ratings[:3])):
    # Only compare on movies both rated
    sim = cosine_similarity(dave, user_vec)
    print(f"  sim(Dave, {name}) = {sim:.4f}")

# Recommend movies Dave hasn't seen, from most similar user (Bob)
bob = ratings[1]
not_seen = np.where(dave == 0)[0]
print(f"\nMovies Dave hasn't seen: {[movies[i] for i in not_seen]}")
print(f"Bob's rating for Interstellar: {bob[1]}  → Recommend to Dave!")

# %% [markdown]
# ## Common Mistakes
#
# ### Mistake 1: Shape Mismatch

# %%
# --- Common Mistakes ---
print("COMMON MISTAKE 1: Shape Confusion")
print("-" * 40)

v_1d = np.array([1, 2, 3])          # shape (3,)
v_2d = np.array([[1], [2], [3]])     # shape (3,1)
print(f"1D vector shape: {v_1d.shape}")
print(f"2D column shape: {v_2d.shape}")

# This silently 'works' but gives outer product, not dot product!
wrong = v_1d * v_2d
print(f"\nv_1d * v_2d (broadcasting — OUTER PRODUCT):\n{wrong}")
print("⚠ This is an outer product, NOT element-wise!")

# The correct approach
correct = np.dot(v_1d, v_2d.flatten())
print(f"\nnp.dot(v_1d, v_2d.flatten()) = {correct}  ✓ Dot product")

print("\nCOMMON MISTAKE 2: Norm vs. Squared Norm")
print("-" * 40)
v = np.array([3.0, 4.0])
norm    = np.linalg.norm(v)           # = 5.0
sq_norm = np.dot(v, v)                # = 25.0  (used in MSE derivatives)
print(f"‖v‖₂           = {norm}  (use for distance)")
print(f"‖v‖₂² = v·v    = {sq_norm}  (use in derivatives, faster)")

print("\nCOMMON MISTAKE 3: In-place vs. Copy")
print("-" * 40)
a = np.array([1, 2, 3])
b = a          # ← DANGEROUS: b is a view!
b[0] = 999
print(f"a after changing b[0]: {a}  ← a was also changed! Use a.copy()")

# %% [markdown]
# ## Interview Questions
#
# **Q1. What is the geometric interpretation of the dot product?**
#
# **A:** $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$.
# It measures the projection of one vector onto another, scaled by their magnitudes.
# When $\theta = 90°$, vectors are orthogonal ($\mathbf{a}\cdot\mathbf{b}=0$).
# In a neural network, the dot product of input and weight vector measures
# how "aligned" the input is with the learned pattern.
#
# **Q2. What is the difference between L1 and L2 norms, and when do you prefer each?**
#
# **A:** L1 ($\sum |v_i|$) produces sparse solutions (many zeros) — used in Lasso
# regularisation for feature selection. L2 ($\sqrt{\sum v_i^2}$) penalises large
# weights smoothly — used in Ridge regularisation to prevent overfitting without
# sparsity.
#
# **Q3. How does cosine similarity differ from Euclidean distance?**
#
# **A:** Cosine similarity measures the **angle** between vectors (ignores magnitude),
# making it ideal for text embeddings where TF-IDF vectors of different document
# lengths should be compared by topic, not word count. Euclidean distance measures
# absolute distance in space — sensitive to scale.
#
# **Q4. Why do we normalise feature vectors before training ML models?**
#
# **A:** Different features have different scales (e.g., age in years vs. income in
# thousands). Without normalisation, gradient descent is skewed toward features with
# large magnitudes. Normalisation ensures all features contribute equally, leading
# to faster, more stable convergence.
#
# **Q5. What does it mean for two vectors to be orthogonal, and where does it appear in ML?**
#
# **A:** Two vectors are orthogonal if their dot product is zero — they are
# geometrically perpendicular. In PCA, the principal components are mutually
# orthogonal (no linear correlation). In neural networks, weight initialisation
# schemes (e.g., orthogonal init) use orthogonal vectors to preserve gradient norms.

# %% [markdown]
# ## Practice Problems
#
# 1. **Norm Comparison**: Given `v = [1, -2, 3, -4, 5]`, compute L1, L2, and L∞ norms.
#    Verify that L∞ ≤ L2 ≤ L1 always holds.
#
# 2. **Unit Vector**: Write a function `unit_vector(v)` that normalises any vector.
#    Handle the zero-vector edge case gracefully.
#
# 3. **Cosine Similarity**: Implement `cosine_similarity(u, v)` from scratch (no sklearn).
#    Test: cos_sim([1,0,0], [1,0,0]) = 1.0, cos_sim([1,0], [0,1]) = 0.0.
#
# 4. **Gradient Step**: Given weights `w = [0.5, -0.3, 0.8]`, gradient `g = [0.2, -0.1, 0.5]`,
#    and learning rate `lr = 0.01`, compute the updated weights after one gradient step.
#
# 5. **Angle Between Vectors**: Write a function that returns the angle (in degrees)
#    between any two non-zero vectors. Verify with [1,0] and [0,1] → 90°.

# %% [markdown]
# ## Solutions

# %%
# --- Solution 1: Norm Comparison ---
print("SOLUTION 1: Norm Comparison")
v = np.array([1, -2, 3, -4, 5], dtype=float)
l1   = np.linalg.norm(v, ord=1)
l2   = np.linalg.norm(v, ord=2)
linf = np.linalg.norm(v, ord=np.inf)
print(f"v = {v}")
print(f"L1  = {l1:.4f}, L2 = {l2:.4f}, L∞ = {linf:.4f}")
print(f"L∞ ≤ L2 ≤ L1: {linf:.4f} ≤ {l2:.4f} ≤ {l1:.4f}  → {linf <= l2 <= l1}")

# --- Solution 2: Unit Vector ---
print("\nSOLUTION 2: Unit Vector")
def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalise the zero vector.")
    return v / norm

print(f"unit_vector([3,4]) = {unit_vector(np.array([3.0,4.0]))}")
print(f"Norm of result     = {np.linalg.norm(unit_vector(np.array([3.0,4.0]))):.6f}")

# --- Solution 3: Cosine Similarity ---
print("\nSOLUTION 3: Cosine Similarity from scratch")
def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

print(f"cos_sim([1,0,0],[1,0,0]) = {cosine_sim(np.array([1,0,0]),np.array([1,0,0])):.4f} (expect 1.0)")
print(f"cos_sim([1,0],[0,1])     = {cosine_sim(np.array([1,0]),np.array([0,1])):.4f} (expect 0.0)")

# --- Solution 4: Gradient Step ---
print("\nSOLUTION 4: Gradient Descent Step")
w  = np.array([0.5, -0.3, 0.8])
g  = np.array([0.2, -0.1, 0.5])
lr = 0.01
w_new = w - lr * g
print(f"w     = {w}")
print(f"g     = {g}")
print(f"w_new = w - lr*g = {w_new}")

# --- Solution 5: Angle Between Vectors ---
print("\nSOLUTION 5: Angle Between Vectors")
def angle_degrees(u: np.ndarray, v: np.ndarray) -> float:
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # numerical safety
    return float(np.degrees(np.arccos(cos_theta)))

print(f"angle([1,0], [0,1])   = {angle_degrees(np.array([1,0]), np.array([0,1])):.2f}°  (expect 90°)")
print(f"angle([1,0], [1,0])   = {angle_degrees(np.array([1,0]), np.array([1,0])):.2f}°  (expect 0°)")
print(f"angle([1,0], [-1,0])  = {angle_degrees(np.array([1,0]), np.array([-1,0])):.2f}° (expect 180°)")

# %% [markdown]
# ## Further Reading
#
# - **3Blue1Brown — Essence of Linear Algebra** (YouTube): Best visual intuition.
#   <https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab>
# - **Gilbert Strang — Introduction to Linear Algebra** (MIT OpenCourseWare)
# - **NumPy Documentation — `numpy.linalg`**:
#   <https://numpy.org/doc/stable/reference/routines.linalg.html>
# - **Andrej Karpathy — Neural Networks: Zero to Hero**:
#   <https://karpathy.ai/zero-to-hero.html>
# - **Mathematics for Machine Learning (Deisenroth, Faisal, Ong)** — free PDF:
#   <https://mml-book.github.io/>
