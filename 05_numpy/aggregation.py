# %% [markdown]
# # NumPy Aggregation Functions
# 
# # Why this matters
# Understanding your data requires summarizing it. In Machine Learning, you often need to find the maximum probability in classification, the mean loss over a batch, or the sum of gradients.
# 
# # Learning Objectives
# - Master basic aggregations: `sum`, `mean`, `max`, `min`.
# - Understand the `axis` parameter thoroughly.
# - Learn to use `keepdims=True`.
# 
# # Concept Explanation
# Aggregation functions reduce the dimensionality of an array.
# - `axis=0`: Aggregate along columns (down the rows).
# - `axis=1`: Aggregate along rows (across the columns).
# - `keepdims=True`: Maintains the original number of dimensions, which is critical for broadcasting afterwards.
# 
# # Beginner Examples
# %%
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Example 1: Global aggregation
print("Global Sum:", np.sum(arr))
print("Global Max:", np.max(arr))

# Example 2: Axis 0 (Column-wise)
print("Column Sum (axis=0):", np.sum(arr, axis=0))

# Example 3: Axis 1 (Row-wise)
print("Row Mean (axis=1):", np.mean(arr, axis=1))

# %% [markdown]
# # Intermediate Examples
# %%
# Example 4: Tabular Data - Feature Means and Variances
# 100 samples, 4 features (e.g., Iris dataset)
X = np.random.rand(100, 4)
feature_means = np.mean(X, axis=0)
feature_stds = np.std(X, axis=0)
print("Feature Means:", feature_means)

# Example 5: Images - Grayscale Conversion
# 10 color images of 32x32 pixels
images = np.random.rand(10, 32, 32, 3)
# Average over the color channels (axis=-1) to convert to grayscale
grayscale = np.mean(images, axis=-1)
print(f"Color shape: {images.shape}, Grayscale shape: {grayscale.shape}")

# Example 6: Classification - Finding the predicted class
# 5 samples, 3 classes (probabilities)
predictions = np.array([[0.1, 0.8, 0.1],
                        [0.9, 0.05, 0.05],
                        [0.2, 0.3, 0.5],
                        [0.7, 0.2, 0.1],
                        [0.3, 0.3, 0.4]])
# Find index of max probability
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted Classes:", predicted_classes)

# %% [markdown]
# # Machine Learning Relevance
# `np.argmax` is used in evaluating classifiers. `np.mean` is used constantly for calculating loss/cost functions across a batch of data.
# 
# # Common Mistakes
# - Confusing `axis=0` and `axis=1`.
# - Forgetting `keepdims=True`, leading to broadcasting errors when you try to subtract the mean from the original array.
# 
# # Interview Questions
# 1. What does `axis=0` mean in `np.sum(matrix, axis=0)`?
# 2. Why is `keepdims=True` useful?
# 3. What is the difference between `np.max` and `np.argmax`?
# 4. How would you calculate the accuracy given an array of true labels and predicted labels?
# 5. How do you find the minimum value of each row in a 2D array?
# 
# # Practice Problems
# 1. Find the sum of all elements in a `3x3` matrix.
# 2. Find the index of the maximum value in a 1D array.
# 3. Given a matrix of student scores `(samples, tests)`, find the average score for each test.
# 4. Given an image batch of shape `(32, 64, 64, 3)`, compute the max pixel value across the whole batch.
# 5. Standardize a matrix `X` by subtracting the column mean and dividing by the column std dev. Use `keepdims`.
# 
# # Solutions
# %%
# Solution 1
mat = np.ones((3, 3))
print("P1:", np.sum(mat))

# Solution 2
arr1d = np.array([1, 5, 9, 2])
print("P2:", np.argmax(arr1d))

# Solution 3
scores = np.random.randint(50, 100, (20, 5))
avg_per_test = np.mean(scores, axis=0)

# Solution 4
batch = np.random.rand(32, 64, 64, 3)
max_val = np.max(batch)

# Solution 5
X = np.random.rand(100, 4)
mean = np.mean(X, axis=0, keepdims=True)
std = np.std(X, axis=0, keepdims=True)
X_std = (X - mean) / std

# %% [markdown]
# # Further Reading
# - NumPy Documentation: `numpy.mean`, `numpy.max`, `numpy.argmin`, `numpy.argmax`.
