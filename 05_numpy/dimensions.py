# %% [markdown]
# # Array Dimensions (Rank): Navigating High-Dimensional Space
# 
# # Why this matters
# Machine learning algorithms frequently operate in high-dimensional spaces. A single feature might be 0D (a scalar), a feature vector is 1D, a batch of feature vectors is 2D, a batch of timeseries data is 3D, and a batch of video data might be 5D! Understanding the concept of "axes" (dimensions) is crucial when computing statistics. For instance, if you have a 2D batch of student scores and you want to calculate the average score per student, you must know whether to apply the mean over `axis=0` or `axis=1`.
# 
# # Learning Objectives
# 1. Understand what dimensions (rank) and axes mean in NumPy.
# 2. Check the number of dimensions using `.ndim`.
# 3. Apply reduction operations (`sum`, `mean`, `max`) across specific axes.
# 4. Understand how data structures map to specific dimensionalities (Scalars, Vectors, Matrices, Tensors).
# 
# # Concept Explanation
# - **Rank:** The number of dimensions of an array. Accessible via `arr.ndim`.
#   - 0D: Scalar (e.g., `5`)
#   - 1D: Vector (e.g., `[1, 2, 3]`)
#   - 2D: Matrix (e.g., `[[1, 2], [3, 4]]`)
#   - 3D+: Tensor (e.g., an RGB image or video data)
# - **Axis:** NumPy operations that collapse arrays (like sum or mean) take an `axis` argument. 
#   - `axis=0` generally points "downwards" (across rows).
#   - `axis=1` generally points "across" (across columns).
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: Inspecting ndim
scalar = np.array(42)
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])
tensor = np.ones((2, 3, 4))

print("Scalar ndim:", scalar.ndim)
print("Vector ndim:", vector.ndim)
print("Matrix ndim:", matrix.ndim)
print("Tensor ndim:", tensor.ndim)

# Example 2: Summing over everything (Default)
# Consider tabular data of 3 samples, 2 features
data = np.array([
    [10, 20],
    [30, 40],
    [50, 60]
])
print("\nGlobal sum:", np.sum(data)) # 10+20+30+40+50+60 = 210

# Example 3: Summing over axis=0 (down the rows)
# Calculates the sum for EACH feature (column)
feature_sums = np.sum(data, axis=0)
print("Sum over axis=0 (feature sums):", feature_sums) # [10+30+50, 20+40+60] = [90, 120]

# Example 4: Summing over axis=1 (across the columns)
# Calculates the sum for EACH sample (row)
sample_sums = np.sum(data, axis=1)
print("Sum over axis=1 (sample sums):", sample_sums) # [10+20, 30+40, 50+60] = [30, 70, 110]

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 5: Finding maximum values along axes
# Useful for finding the highest predicted class in a neural network (argmax)
predictions = np.array([
    [0.1, 0.8, 0.1], # Sample 1: Class 2 has highest probability
    [0.9, 0.05, 0.05], # Sample 2: Class 1 has highest probability
])
# Max probability per sample
max_probs = np.max(predictions, axis=1)
print("\nMax probabilities per sample:", max_probs)

# Index of the max probability per sample (the predicted class!)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes (argmax):", predicted_classes)

# Example 6: Keepdims
# Sometimes you want to retain the original dimension count after a reduction operation
mean_features = np.mean(data, axis=0, keepdims=True)
print("\nMean of features with keepdims=True shape:", mean_features.shape) # (1, 2) instead of (2,)
print("Mean of features:\n", mean_features)

# %% [markdown]
# # Machine Learning Relevance
# In Deep Learning, your data is almost always a Tensor (N-dimensional array).
# For example, training a language model on text: `(batch_size, sequence_length, embedding_dimension)`. This is a 3D tensor.
# If you want to find the average embedding for a word across the batch, you'd calculate `np.mean(tensor, axis=0)`. If you want to average the embeddings across the sequence to get a single document vector, you'd calculate `np.mean(tensor, axis=1)`. Using the wrong axis means mixing samples together, fundamentally ruining your data.
# 
# # Common Mistakes
# 1. **Confusing axis=0 and axis=1:** Beginners often think `axis=0` means "operate on rows" (producing a column vector). It actually means "collapse the rows," sliding down the columns, which produces a row vector of column statistics.
# 2. **Forgetting `keepdims=True`:** When standardizing data (e.g., `(data - mean) / std`), doing `mean = np.mean(data, axis=1)` yields a 1D array. Trying to subtract this 1D array from a 2D matrix can cause broadcasting errors. Using `keepdims=True` maintains the 2D structure `(N, 1)` ensuring smooth broadcasting.
# 
# # Interview Questions
# 1. **What is a Tensor in the context of NumPy/Machine Learning?**
#    *Answer:* A Tensor is simply an N-dimensional array. A 0D tensor is a scalar, 1D is a vector, 2D is a matrix, and 3D+ are higher-order tensors.
# 2. **If you have a matrix of shape `(100, 50)` and you apply `np.mean(matrix, axis=0)`, what is the shape of the output?**
#    *Answer:* The shape will be `(50,)`. `axis=0` collapses the 100 rows, leaving 50 column means.
# 3. **What does the `keepdims` argument do in functions like `np.sum`?**
#    *Answer:* It prevents the reduced dimension from being completely removed. Instead of dropping the dimension, it sets its size to 1. E.g., summing a `(3, 4)` matrix over `axis=1` yields `(3,)` normally, but `(3, 1)` with `keepdims=True`.
# 4. **How would you find the indices of the maximum values in a 1D array?**
#    *Answer:* Using `np.argmax(arr)`.
# 5. **You have RGB image data shaped `(batch_size, height, width, channels)`. You want to convert these to grayscale by averaging the RGB channels. Which axis should you average over?**
#    *Answer:* `axis=3` (or `axis=-1`), because the channels are located in the 4th dimension.
# 
# # Practice Problems
# 1. Create a 3D array of shape `(2, 3, 4)` populated with random numbers. Check its `ndim`.
# 2. Create a `(5, 4)` matrix. Calculate the sum of each row.
# 3. Using the matrix from Problem 2, calculate the minimum value of each column.
# 4. Given predictions of shape `(10, 3)` (10 samples, 3 classes), find the predicted class index for each sample.
# 5. Calculate the mean of the `(5, 4)` matrix from Problem 2 along `axis=1`, keeping the dimensions.
# 
# # Solutions
# 
# %%
# Solution 1
arr_3d = np.random.rand(2, 3, 4)
print("Q1 ndim:", arr_3d.ndim)

# Solution 2
mat = np.arange(20).reshape(5, 4)
row_sums = np.sum(mat, axis=1)
print("Q2 row sums:", row_sums)

# Solution 3
col_mins = np.min(mat, axis=0)
print("Q3 col mins:", col_mins)

# Solution 4
preds = np.random.rand(10, 3)
class_indices = np.argmax(preds, axis=1)
print("Q4 predicted classes:", class_indices)

# Solution 5
mean_keep = np.mean(mat, axis=1, keepdims=True)
print("Q5 shape with keepdims:", mean_keep.shape)

# %% [markdown]
# # Further Reading
# - NumPy Axes Explained: https://numpy.org/doc/stable/glossary.html#term-axis
