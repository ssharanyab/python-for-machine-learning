# %% [markdown]
# # NumPy Random Numbers
# 
# # Why this matters
# Randomness is at the core of Machine Learning. It's used for weight initialization in neural networks, shuffling data, splitting datasets into train/test sets, and stochastic gradient descent.
# 
# # Learning Objectives
# - Understand the modern `np.random.default_rng()` generator.
# - Generate numbers from various distributions (Uniform, Normal).
# - Perform random shuffling and sampling.
# - Understand seeds for reproducibility.
# 
# # Concept Explanation
# - `np.random.seed` (legacy) vs `np.random.default_rng(seed)` (modern, recommended).
# - Normal distribution (`rng.normal`) vs Uniform distribution (`rng.random`, `rng.integers`).
# - `rng.shuffle` (in-place) vs `rng.permutation` (returns a copy).
# 
# # Beginner Examples
# %%
import numpy as np

# Set seed for reproducibility using modern Generator
rng = np.random.default_rng(seed=42)

# Example 1: Uniform distribution [0.0, 1.0)
unif = rng.random(3)
print("Uniform:", unif)

# Example 2: Normal (Gaussian) distribution (mean=0, std=1)
norm = rng.normal(loc=0.0, scale=1.0, size=(2, 2))
print("\nNormal:\n", norm)

# Example 3: Random integers
ints = rng.integers(low=0, high=10, size=5)
print("\nIntegers:", ints)

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Train/Test Split (Shuffling Tabular Data)
X = np.arange(10).reshape(5, 2)
y = np.array([0, 0, 1, 1, 1])
# Get an array of indices and shuffle them
indices = np.arange(X.shape[0])
rng.shuffle(indices) # In-place
X_shuffled = X[indices]
y_shuffled = y[indices]
print(f"\nShuffled X:\n{X_shuffled}\nShuffled y: {y_shuffled}")

# Example 5: Weight Initialization for Neural Networks
# Xavier/Glorot initialization for a layer with 10 inputs and 5 outputs
n_in, n_out = 10, 5
limit = np.sqrt(6 / (n_in + n_out))
weights = rng.uniform(-limit, limit, size=(n_in, n_out))
print(f"\nWeight shape: {weights.shape}, Min: {weights.min():.2f}, Max: {weights.max():.2f}")

# Example 6: Adding Gaussian Noise to Images
image = np.ones((64, 64)) * 0.5
noise = rng.normal(0, 0.05, image.shape)
noisy_image = np.clip(image + noise, 0, 1)
print(f"\nNoisy image max/min: {noisy_image.max():.2f} / {noisy_image.min():.2f}")

# %% [markdown]
# # Machine Learning Relevance
# Reproducibility is vital. Always use a random seed when developing ML pipelines so that you can compare models fairly.
# 
# # Common Mistakes
# - Using the legacy `np.random` module globally instead of isolated generators, which can lead to hard-to-debug state issues in complex code.
# - Using `shuffle` on multidimensional arrays (it only shuffles along the first axis).
# 
# # Interview Questions
# 1. Why is setting a random seed important in Machine Learning?
# 2. What is the difference between `np.random.shuffle` and `np.random.permutation`?
# 3. How would you draw samples from a uniform distribution between -1 and 1?
# 4. What is the modern way to generate random numbers in NumPy?
# 5. How do you randomly select 3 items from a 1D array without replacement?
# 
# # Practice Problems
# 1. Create a `Generator` object with seed 99.
# 2. Generate an array of 1000 samples from a normal distribution with mean 50 and std 10.
# 3. Randomly select 5 unique indices from an array of size 100 using `rng.choice`.
# 4. Create a 3x3 matrix of random integers between 1 and 100.
# 5. Shuffle the rows of a `(10, 4)` matrix without breaking the relationships within the rows.
# 
# # Solutions
# %%
# Solution 1
rng = np.random.default_rng(99)

# Solution 2
samples = rng.normal(50, 10, 1000)

# Solution 3
choices = rng.choice(100, size=5, replace=False)

# Solution 4
mat = rng.integers(1, 101, size=(3, 3))

# Solution 5
X = np.arange(40).reshape(10, 4)
rng.shuffle(X) # Shuffles only axis 0 (rows)

# %% [markdown]
# # Further Reading
# - NumPy Documentation: `numpy.random`.
