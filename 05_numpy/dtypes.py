# %% [markdown]
# # NumPy Data Types (dtypes): Managing Precision and Memory
# 
# # Why this matters
# In machine learning, memory consumption and computational speed are critical constraints. The choice of numerical data type (e.g., 64-bit float vs 32-bit float vs 8-bit integer) directly impacts both. For example, neural network weights are often stored as `float32` rather than `float64` to halve memory usage and double GPU throughput, a practice central to modern deep learning. Image data is typically stored as `uint8` (0-255) to save space before being normalized to floats for training.
# 
# # Learning Objectives
# 1. Understand what a NumPy `dtype` is.
# 2. Learn how to specify and check data types of arrays.
# 3. Cast arrays from one data type to another (`astype`).
# 4. Recognize the memory implications of different data types in ML context.
# 
# # Concept Explanation
# Every NumPy array has an associated `dtype` object that describes the format of the data in the array. Common dtypes include:
# - `np.int8`, `np.int16`, `np.int32`, `np.int64`: Signed integers.
# - `np.uint8`, `np.uint16`: Unsigned integers (great for images).
# - `np.float16`, `np.float32`, `np.float64`: Floating point numbers. `float32` is the workhorse of deep learning.
# - `np.bool_`: Booleans (True/False).
# Choosing the right dtype ensures you do not waste memory or lose precision unnecessarily.
# 
# # Beginner Examples
# 
# %%
import numpy as np

# Example 1: Default Data Types
# Integers usually default to int32 or int64 depending on the OS architecture
arr_int = np.array([1, 2, 3])
print("arr_int dtype:", arr_int.dtype)

# Floats usually default to float64
arr_float = np.array([1.0, 2.5, 3.1])
print("arr_float dtype:", arr_float.dtype)

# Example 2: Specifying Data Types at Creation
# Creating a feature vector for a neural net (float32 is preferred)
features = np.array([0.5, -1.2, 3.4], dtype=np.float32)
print("\nfeatures dtype:", features.dtype)

# Example 3: Creating Image Data (Pixels are usually 0-255 unsigned 8-bit integers)
image_pixels = np.array([255, 128, 0], dtype=np.uint8)
print("\nimage_pixels dtype:", image_pixels.dtype)
print("image_pixels:", image_pixels)

# %% [markdown]
# # Intermediate Examples
# 
# %%
# Example 4: Casting arrays with astype()
# Normalizing image pixels to 0.0 - 1.0 (requires converting to float first)
image_pixels_float = image_pixels.astype(np.float32) / 255.0
print("Normalized pixels:", image_pixels_float)
print("Normalized dtype:", image_pixels_float.dtype)

# Example 5: Upcasting behavior
# Mixing ints and floats will upcast everything to float
mixed_arr = np.array([1, 2.5, 3])
print("\nmixed_arr dtype:", mixed_arr.dtype)

# Example 6: Memory Size Inspection
# A tabular dataset of 1 million rows, 10 columns
num_elements = 1_000_000 * 10
arr_f64 = np.ones(num_elements, dtype=np.float64)
arr_f32 = np.ones(num_elements, dtype=np.float32)

print(f"\nSize in bytes (float64): {arr_f64.nbytes / 1e6} MB")
print(f"Size in bytes (float32): {arr_f32.nbytes / 1e6} MB")

# Example 7: Overflow warning with small types
# uint8 goes from 0 to 255. Adding to 255 wraps around!
small_int = np.array([250, 255], dtype=np.uint8)
print("\nBefore overflow:", small_int)
print("After + 10:", small_int + 10)  # 255 + 10 = 9 (wraps around)

# %% [markdown]
# # Machine Learning Relevance
# Memory bandwidth is frequently the bottleneck in training AI models. Downcasting data from `float64` (default in pandas/numpy) to `float32` (or even `float16` in Mixed Precision Training) can drastically reduce memory footprint and speed up operations on GPUs. In Natural Language Processing, tokens/vocabulary indices are stored as integers (`int32` or `int64`). Images in disk and RAM are kept as `uint8` and only cast to `float32` in batches during training.
# 
# # Common Mistakes
# 1. **Unintentional Float64:** Using `np.zeros()` creates `float64` by default. If feeding to a PyTorch network expecting `float32`, it will crash or implicitly cast, slowing things down.
# 2. **Overflows in uint8:** When augmenting images (e.g., adding brightness), doing `image + 20` when `image` is `uint8` might cause bright pixels to wrap around to black (0).
# 3. **Precision Loss:** Casting large numbers to `float16` might result in `inf` (infinity) due to its limited dynamic range.
# 
# # Interview Questions
# 1. **Why do deep learning frameworks like PyTorch and TensorFlow prefer `float32` over `float64`?**
#    *Answer:* `float32` uses half the memory of `float64`, doubling the number of weights you can fit in VRAM and significantly speeding up matrix multiplications on GPUs, with negligible loss in model performance.
# 2. **What happens if you have an array `a = np.array([255], dtype=np.uint8)` and you compute `a + 1`?**
#    *Answer:* It results in `[0]` due to integer overflow, because 255 is the maximum value represented by an 8-bit unsigned integer.
# 3. **How do you change the data type of an existing NumPy array?**
#    *Answer:* Using the `.astype()` method, e.g., `arr.astype(np.float32)`.
# 4. **How can you find out how much memory an array occupies?**
#    *Answer:* By checking the `arr.nbytes` attribute.
# 5. **If you perform an operation between an `int32` array and a `float32` array, what is the resulting dtype?**
#    *Answer:* NumPy will upcast to `float64` (on standard systems) or `float32` to avoid data loss. Generally, operations between int and float yield floats.
# 
# # Practice Problems
# 1. Create an array of integers `[10, 20, 30]` and explicitly make it `int16`.
# 2. Create an array of random values between 0 and 1, and ensure it is `float32`.
# 3. Convert an array of boolean values `[True, False, True]` to an integer array.
# 4. Create an array of shape (100, 100) filled with 255, using `uint8`, and calculate its size in bytes.
# 5. Demonstrate integer overflow by creating a `int8` array with the value `127` and adding `1` to it.
# 
# # Solutions
# 
# %%
# Solution 1
arr1 = np.array([10, 20, 30], dtype=np.int16)
print("Q1:", arr1.dtype)

# Solution 2
arr2 = np.random.rand(5).astype(np.float32)
print("Q2:", arr2.dtype)

# Solution 3
arr3_bool = np.array([True, False, True])
arr3_int = arr3_bool.astype(np.int8)
print("Q3:", arr3_int)

# Solution 4
arr4 = np.full((100, 100), 255, dtype=np.uint8)
print("Q4 bytes:", arr4.nbytes)

# Solution 5
arr5 = np.array([127], dtype=np.int8)
print("Q5 (overflow):", arr5 + 1) # Expected: -128

# %% [markdown]
# # Further Reading
# - NumPy Data Types: https://numpy.org/doc/stable/user/basics.types.html
# - Mixed Precision Training (NVIDIA): https://developer.nvidia.com/automatic-mixed-precision
