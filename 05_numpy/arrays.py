# %% [markdown]
# # NumPy Arrays: The Foundation of Machine Learning Data
# 
# # Why this matters
# In machine learning, data is represented numerically. Whether you are dealing with a 28x28 pixel grayscale image, a tabular dataset of housing prices, or a sequence of words encoded as a feature vector, everything is ultimately stored as an array of numbers. NumPy arrays (`ndarray`) provide a highly optimized, memory-efficient, and fast way to store and manipulate these numerical representations, forming the backbone of almost all ML frameworks in Python (like PyTorch, TensorFlow, and Scikit-Learn).
# 
# # Learning Objectives
# 1. Understand what a NumPy `ndarray` is and how it differs from a Python list.
# 2. Create arrays from scratch and from existing lists.
# 3. Use specialized array creation functions (zeros, ones, arange, linspace).
# 4. Represent tabular data, feature vectors, and images using NumPy arrays.
# 
# # Concept Explanation
# A NumPy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.
# Python lists can hold heterogeneous data types and are stored as arrays of pointers, making them slow and memory-inefficient for large mathematical operations. NumPy arrays store elements contiguously in memory, enabling vectorized operations written in highly optimized C and Fortran code.
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: Creating a 1D array (Feature Vector)
# Representing a single home's features: [bedrooms, bathrooms, sqft, age]
house_features = np.array([3, 2, 1500, 10])
print("House Features:", house_features)
print("Type:", type(house_features))

# Example 2: Creating a 2D array (Tabular Data)
# Multiple houses
house_data = np.array([
    [3, 2, 1500, 10],
    [4, 3, 2500, 5],
    [2, 1, 900, 45]
])
print("\nHouse Data:\n", house_data)

# Example 3: Creating an array of zeros (Placeholder for predictions)
predictions = np.zeros(5)
print("\nPredictions (zeros):", predictions)

# Example 4: Creating an array of ones (Initialization)
weights = np.ones(4)
print("\nWeights (ones):", weights)

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 5: Creating arrays with numerical ranges (Useful for plotting functions)
x_values = np.arange(0, 10, 2)  # Start, Stop (exclusive), Step
print("Arange:", x_values)

# Example 6: Generating evenly spaced numbers over an interval (Useful for evaluating functions)
x_lin = np.linspace(0, 1, 5) # Start, Stop (inclusive), Num points
print("\nLinspace:", x_lin)

# Example 7: Creating arrays with random values (Weights initialization in Neural Nets)
np.random.seed(42) # For reproducibility
random_weights = np.random.rand(3, 3)
print("\nRandom Weights (Uniform 0-1):\n", random_weights)

# Example 8: Creating an Identity matrix
identity_matrix = np.eye(3)
print("\nIdentity Matrix:\n", identity_matrix)

# Example 9: Generating normally distributed data (Standard Normal)
normal_data = np.random.randn(5)
print("\nNormally distributed data:\n", normal_data)

# %% [markdown]
# # Machine Learning Relevance
# In Computer Vision, a color image is typically represented as a 3D NumPy array of shape `(height, width, channels)`. For instance, a 1920x1080 RGB image is an array of shape `(1080, 1920, 3)`.
# In Natural Language Processing, text might be mapped to 1D vectors (embeddings) of size e.g. 768. 
# Without the contiguous memory layout and vectorized operations provided by arrays, training an ML model would take exponentially longer.
# 
# # Common Mistakes
# 1. **Appending to Arrays:** Unlike Python lists, NumPy arrays have a fixed size upon creation. Appending to a NumPy array creates a completely new array, which is very inefficient in loops.
# 2. **Mixing Data Types:** Putting strings and integers in a NumPy array will upcast everything to strings, destroying mathematical capability.
# 3. **Not Pre-allocating Memory:** Using lists to collect data and then converting to a NumPy array at the end is fine for small data, but for large loops, it's better to initialize an empty NumPy array (`np.zeros`) and fill it.
# 
# # Interview Questions
# 1. **What is the fundamental difference between a Python list and a NumPy array in terms of memory allocation?**
#    *Answer:* Python lists contain pointers to objects scattered in memory, whereas NumPy arrays store homogeneous elements contiguously in a single block of memory, allowing for fast CPU cache access and vectorized operations.
# 2. **How do you generate an array of 100 evenly spaced numbers between 0 and 1?**
#    *Answer:* `np.linspace(0, 1, 100)`
# 3. **Why shouldn't you dynamically append elements to a NumPy array in a loop?**
#    *Answer:* Because NumPy arrays are fixed-size in memory. Appending forces NumPy to allocate a new block of memory and copy all elements over, resulting in O(N^2) time complexity.
# 4. **How do you create a 3x3 matrix initialized with random values from a standard normal distribution?**
#    *Answer:* `np.random.randn(3, 3)`
# 5. **If you pass a nested list `[[1,2], [3,4.5]]` to `np.array()`, what will the data type of the resulting array be?**
#    *Answer:* The integers will be upcast to floats to maintain a homogeneous data type.
# 
# # Practice Problems
# 1. Create a 1D array representing a feature vector of 5 random floats between 0 and 1.
# 2. Create a 2D array of shape (4, 3) containing entirely of zeros to represent a batch of 4 empty images of 3 pixels each.
# 3. Create a 3x3 identity matrix and multiply it by 5.
# 4. Generate an array of numbers from 10 to 50 with a step of 5.
# 5. Create an array of 50 evenly spaced numbers from 0 to 2*PI.
# 
# # Solutions
# 
# %%
# Solution 1
feature_vector = np.random.rand(5)
print("Q1:", feature_vector)

# Solution 2
empty_images = np.zeros((4, 3))
print("Q2:\n", empty_images)

# Solution 3
scaled_identity = np.eye(3) * 5
print("Q3:\n", scaled_identity)

# Solution 4
stepped_array = np.arange(10, 50, 5)
print("Q4:", stepped_array)

# Solution 5
angles = np.linspace(0, 2 * np.pi, 50)
print("Q5:", angles[:5], "...")

# %% [markdown]
# # Further Reading
# - NumPy Official Documentation: https://numpy.org/doc/stable/
# - Contiguous Memory Allocation: https://en.wikipedia.org/wiki/Contiguous_memory_allocation
