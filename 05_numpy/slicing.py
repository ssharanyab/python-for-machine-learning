# %% [markdown]
# # Array Slicing: Efficient Sub-array Extraction
# 
# # Why this matters
# While indexing pulls out specific elements, slicing pulls out continuous chunks (sub-arrays) of data. In machine learning, dataset splitting (Train vs. Test split) and sequence batching (e.g., getting the first 100 frames of a video or a chunk of audio) rely heavily on slicing. Furthermore, NumPy slicing returns a "view" of the original array, meaning it uses zero extra memory and is computationally instantaneous, which is critical when handling gigabytes of data.
# 
# # Learning Objectives
# 1. Master slicing syntax `start:stop:step` for 1D arrays.
# 2. Slice multi-dimensional arrays effectively (e.g., extracting columns).
# 3. Understand the crucial concept of NumPy "Views" vs "Copies".
# 
# # Concept Explanation
# Slicing syntax is `arr[start:stop:step]`.
# - `start` is inclusive.
# - `stop` is exclusive.
# - If omitted, `start` defaults to 0, `stop` defaults to the end, and `step` defaults to 1.
# - Slicing a NumPy array returns a **View**, meaning the new array shares the same memory block as the original array. Modifying the slice modifies the original data.
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: 1D Slicing
timeseries = np.array([10, 15, 20, 25, 30, 35, 40])
print("First 3 elements:", timeseries[:3])
print("From index 3 to end:", timeseries[3:])
print("Every other element:", timeseries[::2])
print("Reversed array:", timeseries[::-1])

# Example 2: 2D Slicing (Crucial for ML feature/label separation)
# Imagine 4 samples. Column 0,1 are features. Column 2 is the label.
dataset = np.array([
    [1.5, 2.3, 0],
    [4.1, 5.5, 1],
    [2.2, 3.1, 0],
    [9.0, 8.8, 1]
])

# Extracting Features (All rows, Columns 0 and 1)
X = dataset[:, :2] # ':' means "all rows". ':2' means up to col 2 (exclusive)
print("\nFeatures (X):\n", X)

# Extracting Labels (All rows, Last column)
y = dataset[:, -1]
print("\nLabels (y):", y)

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 3: Slicing an Image (Cropping)
# Create a dummy 10x10 grayscale image
image = np.arange(100).reshape(10, 10)
# Crop the center 4x4 area (rows 3 to 7, cols 3 to 7)
crop = image[3:7, 3:7]
print("\nCropped 4x4 image:\n", crop)

# Example 4: The Power and Danger of Views
original_arr = np.array([1, 2, 3, 4, 5])
slice_arr = original_arr[1:4] # View of [2, 3, 4]
print("\nOriginal before:", original_arr)

# Modifying the slice!
slice_arr[0] = 99
print("Slice after modification:", slice_arr)
print("Original AFTER modification:", original_arr) # The original changed too!

# Example 5: Forcing a Copy
original_arr2 = np.array([1, 2, 3, 4, 5])
copy_arr = original_arr2[1:4].copy() # Explicitly request a copy
copy_arr[0] = 99
print("\nOriginal stays unchanged if copy is used:", original_arr2)

# %% [markdown]
# # Machine Learning Relevance
# **Train-Test Split:** If you have an array of 1000 samples, you might use slicing to split them:
# `train_X = X[:800, :]`
# `test_X = X[800:, :]`
# 
# **Image Processing:** Data augmentation often involves cropping. `img[10:110, 10:110, :]` crops a 100x100 patch from an image.
# 
# **Time Series / NLP:** When processing sequences, you might pass sliding windows over data. Slicing handles this effortlessly and with zero memory overhead due to views.
# 
# # Common Mistakes
# 1. **Accidental Modification:** Forgetting that slices are views is a massive source of bugs. If you slice a dataset, manipulate the slice, and your original dataset gets corrupted, this is why. Always use `.copy()` if you intend to modify the slice independently.
# 2. **Off-by-One Errors:** Remembering that `start` is inclusive and `stop` is exclusive. `arr[0:5]` returns 5 elements (indices 0,1,2,3,4).
# 3. **Confusing List Slicing with Array Slicing:** Python lists don't allow multidimensional slicing like `list[:, 1]`. Only NumPy arrays support this syntax.
# 
# # Interview Questions
# 1. **What is a "View" in NumPy and how does it differ from a "Copy"?**
#    *Answer:* A view is an array that shares the same underlying data buffer as the original array. Modifying the view modifies the original. A copy is a completely independent duplicate of the data in memory.
# 2. **Does slicing a NumPy array return a view or a copy?**
#    *Answer:* Slicing `[::]` returns a view. (Contrast this with boolean indexing and fancy indexing, which return copies).
# 3. **How do you select only the last column of a 2D array `X`?**
#    *Answer:* `X[:, -1]`. Note that this returns a 1D array.
# 4. **How do you select the last column of a 2D array `X` but keep it as a 2D column vector?**
#    *Answer:* `X[:, -1:]`.
# 5. **How would you reverse the rows of a 2D matrix?**
#    *Answer:* `matrix[::-1, :]`.
# 
# # Practice Problems
# 1. Create an array 0-9. Slice out the elements `[4, 5, 6]`.
# 2. Create a 5x5 matrix. Extract the top-right 3x3 sub-matrix.
# 3. Given a matrix of shape `(100, 10)`, slice out the first 80 rows for training, and the remaining 20 rows for testing.
# 4. Create an array 0-5. Create a slice of the first 3 elements. Modify the first element of the slice to be 100. Print the original array to verify it changed.
# 5. Extract all rows but only the even-indexed columns (0, 2, 4...) from a 2D matrix.
# 
# # Solutions
# 
# %%
# Solution 1
arr = np.arange(10)
print("Q1 slice:", arr[4:7])

# Solution 2
mat = np.arange(25).reshape(5, 5)
top_right = mat[:3, 2:]
print("Q2 top right:\n", top_right)

# Solution 3
data = np.zeros((100, 10))
X_train = data[:80, :]
X_test = data[80:, :]
print("Q3 train shape:", X_train.shape, "test shape:", X_test.shape)

# Solution 4
arr4 = np.arange(6)
slc = arr4[:3]
slc[0] = 100
print("Q4 original changed:", arr4)

# Solution 5
mat5 = np.arange(16).reshape(4, 4)
even_cols = mat5[:, ::2]
print("Q5 even cols:\n", even_cols)

# %% [markdown]
# # Further Reading
# - NumPy Views vs Copies: https://numpy.org/doc/stable/user/basics.copies.html
