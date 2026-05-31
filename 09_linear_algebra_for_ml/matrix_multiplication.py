# %% [markdown]
# # Matrix Multiplication in Linear Algebra for Machine Learning
#
# **Module**: 09 — Linear Algebra for ML
# **File**: matrix_multiplication.py
# **Format**: Jupytext Light (compatible with Jupyter via `jupytext`)

# %% [markdown]
# ## Why This Matters
#
# **Matrix multiplication is the most computationally intensive operation in all of
# deep learning.** Every single forward pass and backward pass in a neural network
# is fundamentally a series of matrix multiplications.
#
# | Operation | Matrix Multiplication |
# |---|---|
# | Neural network forward pass | $Z = XW^T + b$ |
# | Batched predictions | $\hat{Y} = XW$ |
# | Attention in Transformers | $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$ |
# | PCA projection | $Z = XP$ where $P$ = eigenvectors |
# | Linear regression (Normal Eq) | $(X^TX)^{-1}X^Ty$ |
#
# Modern GPUs are essentially matrix-multiplication accelerators.
# Understanding this operation unlocks understanding of all deep learning.

# %% [markdown]
# ## Learning Objectives
#
# By the end of this notebook you will be able to:
# 1. Explain the mechanics of matrix multiplication (row · column dot products)
# 2. State and apply shape rules: $(m\times k)(k\times n) = m\times n$
# 3. Verify non-commutativity and distributivity of matrix products
# 4. Implement matrix multiplication from scratch and via NumPy
# 5. Understand batch matrix multiplication
# 6. Implement a full neural network forward pass using matrix multiplication
# 7. Explain the Transformer attention mechanism as matrix operations

# %% [markdown]
# ## Concept Explanation
#
# ### Definition
#
# For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:
#
# $$(AB)_{ij} = \sum_{l=1}^{k} A_{il} \cdot B_{lj}$$
#
# The $(i,j)$ element of the product is the **dot product** of
# row $i$ of $A$ with column $j$ of $B$.
#
# ### Shape Rule (the critical constraint)
#
# ```
# A         ×   B         =  C
# (m × k)   ×   (k × n)  =  (m × n)
#                 ↑ must match
# ```
#
# ### Properties
#
# | Property | Formula | Note |
# |---|---|---|
# | Associative | $(AB)C = A(BC)$ | Grouping doesn't matter |
# | Distributive | $A(B+C) = AB + AC$ | Standard |
# | NOT commutative | $AB \neq BA$ in general | Critical! |
# | Transpose rule | $(AB)^T = B^T A^T$ | Reversed order |
# | Identity | $IA = AI = A$ | $I$ = identity matrix |

# %%
import numpy as np
import matplotlib.pyplot as plt
import time

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
# ### Example 1: Manual Matrix Multiplication

# %%
# --- Example 1: Matrix Multiplication — Step by Step ---
print("=" * 60)
print("EXAMPLE 1: Manual Matrix Multiplication — Step by Step")
print("=" * 60)

A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])    # shape (3, 2)

print(f"A shape: {A.shape}  (m=2, k=3)")
print(f"B shape: {B.shape}  (k=3, n=2)")
print(f"Expected output shape: (2, 2)\n")

# Manual computation
print("Computing C = A @ B element by element:")
C = np.zeros((2, 2))
for i in range(A.shape[0]):       # rows of A
    for j in range(B.shape[1]):   # cols of B
        dot = 0
        for l in range(A.shape[1]):  # shared dimension k
            dot += A[i, l] * B[l, j]
        C[i, j] = dot
        print(f"  C[{i},{j}] = row {i} of A · col {j} of B = {A[i,:]} · {B[:,j]} = {dot}")

print(f"\nC (manual):\n{C.astype(int)}")
print(f"C (numpy @):\n{A @ B}")
print(f"Match: {np.allclose(C, A @ B)}")

# %% [markdown]
# ### Example 2: Shape Rules and Common Errors

# %%
# --- Example 2: Shape Rules ---
print("=" * 60)
print("EXAMPLE 2: Shape Rules")
print("=" * 60)

def show_matmul_shape(shape_A, shape_B):
    try:
        A = np.random.randn(*shape_A)
        B = np.random.randn(*shape_B)
        C = A @ B
        print(f"  {shape_A} @ {shape_B} → {C.shape}  ✓")
    except ValueError as e:
        print(f"  {shape_A} @ {shape_B} → ERROR: {e}  ✗")

valid_pairs = [
    ((2, 3), (3, 4)),
    ((5, 1), (1, 7)),
    ((100, 256), (256, 10)),   # typical NN layer: 100 samples, 256→10
    ((3, 3), (3,)),            # matrix × vector
]
invalid_pairs = [
    ((2, 3), (2, 4)),  # inner dims mismatch
    ((3, 4), (3, 2)),  # inner dims mismatch
]

print("Valid multiplications:")
for sa, sb in valid_pairs:
    show_matmul_shape(sa, sb)

print("\nInvalid multiplications:")
for sa, sb in invalid_pairs:
    show_matmul_shape(sa, sb)

# %% [markdown]
# ### Example 3: Non-commutativity

# %%
# --- Example 3: Non-commutativity ---
print("=" * 60)
print("EXAMPLE 3: Non-Commutativity of Matrix Multiplication")
print("=" * 60)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

AB = A @ B
BA = B @ A
print(f"A =\n{A}\n")
print(f"B =\n{B}\n")
print(f"AB =\n{AB}\n")
print(f"BA =\n{BA}\n")
print(f"AB == BA: {np.allclose(AB, BA)}  ← Almost never equal!")
print(f"AB - BA =\n{AB - BA}")

# This means ORDER MATTERS in gradient computations!
print("\n⚠ In backpropagation, the order of matrix multiplications")
print("   in the chain rule is critical. Reversing it gives wrong gradients.")

# %% [markdown]
# ### Example 4: Matrix × Vector (the fundamental NN operation)

# %%
# --- Example 4: Matrix-Vector Multiplication ---
print("=" * 60)
print("EXAMPLE 4: Matrix × Vector — One Neural Network Neuron")
print("=" * 60)

# A single neuron: z = w^T x + b
x = np.array([1.5, 0.8, -0.3, 2.1])   # input feature vector, shape (4,)
W = np.random.randn(3, 4)              # weight matrix (3 neurons, 4 inputs)
b = np.array([0.1, -0.2, 0.3])        # bias vector

z = W @ x + b
print(f"Input x (4 features):   {x}")
print(f"Weight W (3×4):\n{W.round(3)}")
print(f"Bias b (3):  {b}")
print(f"\nz = Wx + b = {z.round(4)}   shape: {z.shape}")
print(f"\nEach element of z is one neuron's pre-activation.")

# Apply ReLU activation
relu = np.maximum(0, z)
print(f"a = ReLU(z) = {relu.round(4)}")

# %% [markdown]
# ### Example 5: Transposed Multiplication Identities

# %%
# --- Example 5: Transpose Identity (AB)^T = B^T A^T ---
print("=" * 60)
print("EXAMPLE 5: Transpose Identity (AB)^T = B^T A^T")
print("=" * 60)

A = np.random.randn(3, 5)
B = np.random.randn(5, 2)

AB = A @ B
left  = AB.T
right = B.T @ A.T

print(f"A: {A.shape}, B: {B.shape}")
print(f"(AB)^T:       {left.shape}")
print(f"B^T A^T:      {right.shape}")
print(f"Are equal:    {np.allclose(left, right)}")
print("\nThis identity is CRITICAL in backpropagation:")
print("  dL/dW = X^T dL/dZ   (requires transposing activations)")

# %% [markdown]
# ## Intermediate Examples
#
# ### Example 6: Batched Matrix Multiplication

# %%
# --- Example 6: Batched Matrix Multiplication ---
print("=" * 60)
print("EXAMPLE 6: Batched Matrix Multiplication (Training in Batches)")
print("=" * 60)

# In practice, we process mini-batches of samples simultaneously.
# X shape: (batch_size, n_features)
# W shape: (n_features, n_outputs)

np.random.seed(42)
batch_size, n_features, n_outputs = 32, 128, 10

X_batch = np.random.randn(batch_size, n_features)
W       = np.random.randn(n_features, n_outputs)
b       = np.random.randn(n_outputs)

# One call computes ALL 32 predictions in parallel!
Z = X_batch @ W + b    # shape (32, 10)

print(f"Batch X shape:  {X_batch.shape}")
print(f"Weight W shape: {W.shape}")
print(f"Bias b shape:   {b.shape}")
print(f"Output Z shape: {Z.shape}  ← 32 predictions × 10 classes")
print(f"\nFirst prediction: {Z[0].round(3)}")

# Compare to loop (same result, much slower)
start = time.perf_counter()
for _ in range(1000):
    _ = X_batch @ W + b
t_vectorised = (time.perf_counter() - start) * 1000

start = time.perf_counter()
for _ in range(1000):
    Z_loop = np.zeros((batch_size, n_outputs))
    for i in range(batch_size):
        Z_loop[i] = X_batch[i] @ W + b
t_loop = (time.perf_counter() - start) * 1000

print(f"\nSpeed comparison (1000 iterations × batch_size={batch_size}):")
print(f"  Vectorised:  {t_vectorised:.2f} ms")
print(f"  Loop:        {t_loop:.2f} ms")
print(f"  Speedup:     {t_loop/t_vectorised:.1f}×")

# %% [markdown]
# ### Example 7: Gram Matrix

# %%
# --- Example 7: Gram Matrix X X^T ---
print("=" * 60)
print("EXAMPLE 7: Gram Matrix — Kernel Methods & Style Transfer")
print("=" * 60)

# The Gram matrix G = X X^T captures pairwise similarities between rows.
# G[i,j] = dot product between sample i and sample j = their similarity.

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

G = X @ X.T
print(f"X =\n{X}")
print(f"\nGram matrix G = X X^T:\n{G}")
print(f"\nG[0,0] = ‖x₀‖² = {G[0,0]:.0f}   (self-similarity)")
print(f"G[0,1] = x₀·x₁ = {G[0,1]:.0f}   (cross-sample similarity)")

# In neural style transfer, Gram matrices of CNN feature maps
# capture texture / style information.
print("\nML Applications of Gram Matrices:")
print("  - Kernel SVM: K(xi, xj) = xi · xj  (linear kernel)")
print("  - Neural Style Transfer: Gram of conv feature maps = style")
print("  - Covariance: Gram / (n-1) when data is zero-centred")

# %% [markdown]
# ## Machine Learning Relevance
#
# ### Example 8: Full 2-Layer Neural Network Forward Pass

# %%
# --- Example 8: 2-Layer Neural Network Forward Pass ---
print("=" * 60)
print("EXAMPLE 8: Neural Network Forward Pass via Matrix Multiplication")
print("=" * 60)

np.random.seed(0)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# Architecture: 4 input → 8 hidden → 3 output (3-class classification)
n, d_in, d_h, d_out = 10, 4, 8, 3

# Kaiming He weight initialisation
W1 = np.random.randn(d_in, d_h) * np.sqrt(2.0 / d_in)
b1 = np.zeros(d_h)
W2 = np.random.randn(d_h, d_out) * np.sqrt(2.0 / d_h)
b2 = np.zeros(d_out)

# Input batch
X = np.random.randn(n, d_in)

print(f"Architecture: {d_in} → {d_h} → {d_out}")
print(f"Batch size: {n}\n")

# ----- Layer 1: Linear + ReLU -----
Z1 = X @ W1 + b1        # shape (n, d_h)
A1 = relu(Z1)            # shape (n, d_h)
print(f"Layer 1: X{X.shape} @ W1{W1.shape} + b1{b1.shape} → Z1{Z1.shape}")
print(f"After ReLU: A1{A1.shape}")

# ----- Layer 2: Linear + Softmax -----
Z2 = A1 @ W2 + b2        # shape (n, d_out)
A2 = softmax(Z2)          # shape (n, d_out) — class probabilities
print(f"\nLayer 2: A1{A1.shape} @ W2{W2.shape} + b2{b2.shape} → Z2{Z2.shape}")
print(f"After Softmax: A2{A2.shape}")
print(f"\nPredicted probabilities (first 3 samples):\n{A2[:3].round(4)}")
print(f"Row sums (should be 1.0): {A2[:3].sum(axis=1)}")

# Parameter count
params_W1 = W1.size
params_b1 = b1.size
params_W2 = W2.size
params_b2 = b2.size
total = params_W1 + params_b1 + params_W2 + params_b2
print(f"\nParameter counts:")
print(f"  W1: {params_W1}, b1: {params_b1}, W2: {params_W2}, b2: {params_b2}")
print(f"  Total parameters: {total}")

# %% [markdown]
# ### Example 9: Transformer Scaled Dot-Product Attention

# %%
# --- Example 9: Scaled Dot-Product Attention (Transformer) ---
print("=" * 60)
print("EXAMPLE 9: Transformer Attention — Pure Matrix Multiplication")
print("=" * 60)

# The attention mechanism that powers GPT, BERT, etc. is:
# Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
# Everything is matrix multiplication!

np.random.seed(7)
seq_len = 5     # sequence length (tokens)
d_model = 8    # embedding dimension
d_k = 4        # key/query dimension

# Linear projections (in practice, these are learned weight matrices)
W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k)

# Input: sequence of token embeddings
X_seq = np.random.randn(seq_len, d_model)

# Project to Q, K, V
Q = X_seq @ W_Q   # (seq_len, d_k)
K = X_seq @ W_K   # (seq_len, d_k)
V = X_seq @ W_V   # (seq_len, d_k)

print(f"Input sequence: {X_seq.shape}  (tokens × embedding_dim)")
print(f"Q, K, V shapes: {Q.shape}")

# Attention scores: Q K^T  scaled
scores = Q @ K.T / np.sqrt(d_k)   # (seq_len, seq_len)
print(f"\nAttention scores (Q K^T / sqrt(d_k)): {scores.shape}")
print(f"Raw scores:\n{scores.round(3)}")

# Softmax over keys dimension
def softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

attn_weights = softmax_rows(scores)   # (seq_len, seq_len)
print(f"\nAttention weights (after softmax):\n{attn_weights.round(3)}")
print(f"Row sums: {attn_weights.sum(axis=1).round(6)}")

# Weighted sum of values
output = attn_weights @ V   # (seq_len, d_k)
print(f"\nAttention output: {output.shape}")
print(f"First token output: {output[0].round(4)}")
print("\nAll 3 projections (Q@K^T and attention@V) are matrix multiplications!")

# %% [markdown]
# ### Example 10: Performance — NumPy vs Pure Python

# %%
# --- Example 10: Performance Benchmarking ---
print("=" * 60)
print("EXAMPLE 10: Performance — Why Matrix Ops Are Vectorised")
print("=" * 60)

sizes = [10, 50, 100, 200, 500]
times_numpy = []
times_python = []

for n in sizes:
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    # NumPy timing
    reps = max(1, int(1000 / n))
    start = time.perf_counter()
    for _ in range(reps):
        C = A @ B
    t_np = (time.perf_counter() - start) / reps * 1000
    times_numpy.append(t_np)

    # Python loop timing (only for small n)
    if n <= 50:
        start = time.perf_counter()
        C_py = [[sum(A[i,k]*B[k,j] for k in range(n)) for j in range(n)] for i in range(n)]
        t_py = (time.perf_counter() - start) * 1000
    else:
        t_py = None
    times_python.append(t_py)

    print(f"  n={n:4d}: NumPy={t_np:.4f}ms" + (f"  Python={t_py:.2f}ms  Speedup={t_py/t_np:.0f}×" if t_py else "  (Python skipped — too slow)"))

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(sizes, times_numpy, "o-", color="#00d4ff", lw=2, label="NumPy @ (BLAS)")
ax.set_xlabel("Matrix size n (n×n × n×n)")
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("Matrix Multiplication Performance: NumPy BLAS vs Pure Python",
             color="white", fontsize=12)
ax.legend(framealpha=0.3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("matmul_performance.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ## Common Mistakes

# %%
print("COMMON MISTAKE 1: Using * instead of @ for matrix multiplication")
print("-" * 55)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A * B (element-wise, NOT matrix mult):\n{A * B}")
print(f"A @ B (true matrix multiplication):\n{A @ B}")

print("\nCOMMON MISTAKE 2: Inner dimensions must match")
print("-" * 55)
A2 = np.random.randn(3, 4)
B2 = np.random.randn(3, 5)  # wrong inner dim
try:
    _ = A2 @ B2
except ValueError as e:
    print(f"Error: {e}")
print(f"Fix: B must have shape (4, ?) — inner dims (4 and 4) must match")

print("\nCOMMON MISTAKE 3: np.dot vs @ for 2D+ arrays")
print("-" * 55)
A3 = np.random.randn(3, 4)
B3 = np.random.randn(4, 2)
print(f"np.dot(A3, B3) and A3 @ B3 are equivalent for 2D: {np.allclose(np.dot(A3, B3), A3 @ B3)}")
print("But for 3D+ (batched), np.dot behaviour differs from @")
print("→ Always prefer @ for matrix multiplication")

print("\nCOMMON MISTAKE 4: Forgetting to transpose for normal equations")
print("-" * 55)
n, d = 10, 3
X4 = np.random.randn(n, d)
y4 = np.random.randn(n)
# Wrong: X4 @ X4  — shape error or wrong result
# Correct:
XtX = X4.T @ X4    # (d, d)
Xty = X4.T @ y4    # (d,)
print(f"X^T X shape: {XtX.shape}  ✓ (must be square for inversion)")
print(f"X^T y shape: {Xty.shape}  ✓ (solution vector)")

# %% [markdown]
# ## Interview Questions
#
# **Q1. What are the shape requirements for matrix multiplication, and why?**
#
# **A:** For $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$,
# the inner dimensions must be equal ($k = k$) because $(AB)_{ij} = \sum_{l=1}^{k} A_{il}B_{lj}$
# — we sum over $k$ paired elements. The result is $m \times n$.
# In a neural network layer $Z = XW$, if $X$ is $(n, d_{in})$ and $W$ is $(d_{in}, d_{out})$,
# the output $Z$ is $(n, d_{out})$ — $n$ samples each mapped to $d_{out}$ activations.
#
# **Q2. Why is matrix multiplication not commutative?**
#
# **A:** $AB$ and $BA$ have different shapes unless both are square, and even for
# square matrices the products differ. Geometrically, applying transformation $A$
# then $B$ (i.e., $BA$) is generally different from applying $B$ then $A$ (i.e., $AB$).
# This matters in backpropagation: $\partial L / \partial X = \partial L / \partial Z \cdot W^T$
# (not $W^T \cdot \partial L / \partial Z$).
#
# **Q3. What is the computational complexity of multiplying two n×n matrices?**
#
# **A:** Naïve algorithm: $O(n^3)$ — $n^2$ output elements each requiring $n$ multiplications.
# Strassen's algorithm: $O(n^{2.807})$. Modern BLAS implementations use highly optimised
# cache-friendly algorithms. For ML, typical sizes are large but rectangular
# (e.g., $1024 \times 768 \times 512$), which GPUs handle in parallel.
#
# **Q4. How does the Transformer attention mechanism use matrix multiplication?**
#
# **A:** Three steps, all matrix multiplications:
# (1) Project inputs: $Q = XW_Q$, $K = XW_K$, $V = XW_V$
# (2) Compute scores: $S = QK^T / \sqrt{d_k}$ — gives an $n\times n$ attention map
# (3) Weighted sum: $\text{Output} = \text{softmax}(S)V$
# The $n \times n$ attention matrix is the quadratic bottleneck that makes
# standard Transformers $O(n^2)$ in sequence length.
#
# **Q5. What is the Normal Equation in linear regression and why does it involve matrix multiplication?**
#
# **A:** The optimal weights minimising MSE are $\hat{w} = (X^TX)^{-1}X^Ty$.
# $X^TX \in \mathbb{R}^{d\times d}$ is the Gram matrix encoding feature correlations.
# $X^Ty \in \mathbb{R}^{d}$ encodes feature-label correlations.
# We solve the linear system $(X^TX)\hat{w} = X^Ty$ using `np.linalg.solve` (numerically
# stable, avoids explicit inversion). Matrix multiplication is central because
# minimising $\|Xw - y\|^2$ leads directly to the matrix equation.

# %% [markdown]
# ## Practice Problems
#
# 1. **From Scratch**: Implement matrix multiplication using only Python loops and NumPy
#    array access. Verify it matches `A @ B` for a 3×4 and 4×2 matrix.
#
# 2. **Layer Dimensions**: Design a 3-layer neural network with input dim=784 (MNIST image),
#    hidden dims=128 and 64, output dim=10. Write out all weight matrix shapes and the
#    output shape for a batch of 32 samples.
#
# 3. **Transpose Identity**: Verify that $(AB)^T = B^T A^T$ for random matrices
#    $A \in \mathbb{R}^{3\times5}$ and $B \in \mathbb{R}^{5\times4}$.
#
# 4. **Gram Matrix**: Given a 2D point cloud of 50 points, compute the Gram matrix
#    and find the two most similar points (highest off-diagonal value).
#
# 5. **Mini Attention**: Implement scaled dot-product attention from scratch for
#    a 4-token sequence with d_model=6 and d_k=3. Print the attention weight matrix.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Matrix Multiplication from Scratch
print("SOLUTION 1: Matrix Multiplication from Scratch")
def matmul_scratch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Inner dims must match"
    m, k = A.shape
    _, n = B.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    return C

A_test = np.random.randn(3, 4)
B_test = np.random.randn(4, 2)
C_scratch = matmul_scratch(A_test, B_test)
C_numpy   = A_test @ B_test
print(f"Scratch matches NumPy: {np.allclose(C_scratch, C_numpy)}")

# Solution 2: Layer Dimensions
print("\nSOLUTION 2: MNIST Network — Weight Shapes")
layers = [(784, 128), (128, 64), (64, 10)]
batch  = 32
x_shape = (batch, 784)
print(f"Input shape: {x_shape}")
for i, (d_in, d_out) in enumerate(layers):
    W_shape = (d_in, d_out)
    b_shape = (d_out,)
    x_shape = (batch, d_out)
    print(f"Layer {i+1}: W{W_shape} + b{b_shape} → output{x_shape}")

# Solution 3: Transpose Identity
print("\nSOLUTION 3: Transpose Identity (AB)^T = B^T A^T")
A_r = np.random.randn(3, 5)
B_r = np.random.randn(5, 4)
lhs = (A_r @ B_r).T
rhs = B_r.T @ A_r.T
print(f"(AB)^T shape: {lhs.shape}, B^T A^T shape: {rhs.shape}")
print(f"Equal: {np.allclose(lhs, rhs)}")

# Solution 4: Gram Matrix
print("\nSOLUTION 4: Gram Matrix — Pairwise Similarities")
np.random.seed(5)
pts = np.random.randn(50, 2)
G = pts @ pts.T
np.fill_diagonal(G, -np.inf)   # ignore self-similarity
i, j = np.unravel_index(G.argmax(), G.shape)
np.fill_diagonal(G, 0)
print(f"Most similar pair: points {i} and {j} with G[i,j]={G[i,j]:.4f}")
print(f"Point {i}: {pts[i].round(3)},  Point {j}: {pts[j].round(3)}")

# Solution 5: Mini Attention
print("\nSOLUTION 5: Scaled Dot-Product Attention")
np.random.seed(9)
seq_len, d_model, d_k = 4, 6, 3
X_att = np.random.randn(seq_len, d_model)
Wq = np.random.randn(d_model, d_k)
Wk = np.random.randn(d_model, d_k)
Wv = np.random.randn(d_model, d_k)
Q5, K5, V5 = X_att @ Wq, X_att @ Wk, X_att @ Wv
scores5 = Q5 @ K5.T / np.sqrt(d_k)
exp_s = np.exp(scores5 - scores5.max(axis=1, keepdims=True))
attn5 = exp_s / exp_s.sum(axis=1, keepdims=True)
out5 = attn5 @ V5
print(f"Attention weights:\n{attn5.round(3)}")
print(f"Row sums: {attn5.sum(axis=1).round(6)}")
print(f"Output shape: {out5.shape}")

# %% [markdown]
# ## Further Reading
#
# - **3Blue1Brown — Matrix Multiplication as Composition**:
#   <https://www.3blue1brown.com/lessons/matrix-multiplication>
# - **Attention Is All You Need (Vaswani et al., 2017)**:
#   <https://arxiv.org/abs/1706.03762>
# - **Stanford CS231n — Neural Networks Notes**:
#   <https://cs231n.github.io/neural-networks-1/>
# - **NumPy matmul docs**:
#   <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>
# - **The Matrix Cookbook** (dense reference for matrix identities):
#   <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>
