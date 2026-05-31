# %% [markdown]
# # Variables and Memory
#
# # Why this matters
# Variables are the fundamental way we store, manipulate, and reference data. Understanding how Python handles variables—specifically that they are references to objects in memory—is crucial for avoiding subtle bugs when handling large ML datasets.
#
# # Learning Objectives
# 1. Understand how to declare and initialize variables in Python.
# 2. Learn the naming conventions (PEP 8) for variables.
# 3. Grasp the concept of variables as references (pointers) to objects.
# 4. Understand garbage collection and memory management at a high level.
#
# # Concept Explanation
# In Python, a variable is not a bucket that holds data; it is a label or reference pointing to an object in memory. When you write `x = 10`, Python creates an integer object `10` in memory and binds the name `x` to it. Variable names must start with a letter or underscore, and can contain letters, numbers, and underscores. Python is dynamically typed, meaning you don't declare a variable's type explicitly.
#
# # Beginner Examples
# Let's create basic variables and explore them.

# %%
# Integer assignment
epochs = 100
print(f"Epochs: {epochs}")

# Floating point assignment
learning_rate = 0.01
print(f"Learning Rate: {learning_rate}")

# String assignment
model_name = "RandomForest"
print(f"Model Name: {model_name}")

# Multiple assignment
x, y, z = 1, 2, 3
print(f"x={x}, y={y}, z={z}")

# Chained assignment
a = b = c = 0
print(f"a={a}, b={b}, c={c}")

# %% [markdown]
# # Intermediate Examples
# Here we explore identity and reference semantics, which is crucial for handling complex structures.

# %%
# Variables point to the same object
list_a = [1, 2, 3]
list_b = list_a  # list_b points to the exact same list in memory
list_b.append(4)

print("list_a:", list_a) # Notice list_a also changed!
print("Are list_a and list_b the same object?", list_a is list_b)

# Using id() to see memory addresses
x = 42
y = 42
# Small integers are cached in Python, so they often share the same ID
print("ID of x:", id(x))
print("ID of y:", id(y))
print("Are x and y the same object?", x is y)

# %% [markdown]
# # Machine Learning Relevance
# In ML, you work with massive DataFrames and matrices. If you misunderstand variable referencing, you might accidentally modify your original training data when you only meant to modify a slice or a copy of it. This causes data leakage or corrupted datasets. Understanding `copy()` versus deep copy vs reference assignment is mandatory.
#
# # Common Mistakes
# 1. Modifying a mutable object through a new variable and being surprised the original changed.
# 2. Using reserved keywords (like `list`, `str`, `dict`) as variable names, shadowing built-in functions.
# 3. Not using descriptive variable names (`x1`, `x2`, `x3` instead of `features`, `labels`, `predictions`).
# 4. Assuming `x = y` creates a full copy of the data.
#
# # Interview Questions
# 1. **Is Python statically or dynamically typed?**
#    *Answer:* Dynamically typed. Variables don't have types; only the objects they point to have types.
# 2. **What does the `is` keyword do compared to `==`?**
#    *Answer:* `is` checks if two variables point to the exact same object in memory (identity), while `==` checks if the objects have the same value (equality).
# 3. **Explain how garbage collection works in Python.**
#    *Answer:* Python uses reference counting primarily. When an object's reference count drops to zero, it is deallocated. It also uses a cyclic garbage collector to detect and break circular references.
# 4. **Why shouldn't you use `list` as a variable name?**
#    *Answer:* Because `list` is a built-in Python function/type. Overriding it means you can no longer use it to cast items to a list in that scope.
# 5. **If `a = [1, 2]` and `b = a`, how do you make `b` a completely independent copy of `a`?**
#    *Answer:* You can use `b = a.copy()`, `b = a[:]`, or `import copy; b = copy.deepcopy(a)` for nested structures.
#
# # Practice Problems
# 1. Assign your favorite ML algorithm to a variable and print it.
# 2. Create a variable `data` holding a list of numbers. Create another variable `data_copy` that holds a true copy of `data`. Modify `data_copy` and prove `data` did not change.
# 3. Write a small script that demonstrates shadowing a built-in function (like `max`) and how it causes an error later. (Comment out the error-causing line after).
# 4. Swap the values of two variables `alpha` and `beta` without using a temporary third variable.
# 5. Use the `id()` function to check if two identical strings created independently point to the same memory location.
#
# # Solutions

# %%
# Solution to Problem 1
favorite_algo = "Gradient Boosting"
print("Favorite Algorithm:", favorite_algo)

# %%
# Solution to Problem 2
data = [10, 20, 30]
data_copy = data.copy() # or data[:]
data_copy.append(40)
print("Original data:", data)
print("Modified copy:", data_copy)

# %%
# Solution to Problem 3
# max = 10  # Shadowing built-in!
# print(max([1, 2, 3])) # This would raise a TypeError: 'int' object is not callable
# del max # Clean up the shadowing

# %%
# Solution to Problem 4
alpha = 5
beta = 10
alpha, beta = beta, alpha
print(f"alpha: {alpha}, beta: {beta}")

# %%
# Solution to Problem 5
str1 = "machine_learning"
str2 = "machine_learning"
print("ID str1:", id(str1))
print("ID str2:", id(str2))
print("Same object?", str1 is str2) # Often True due to string interning

# %% [markdown]
# # Further Reading
# - [Python Official Tutorial - Variables](https://docs.python.org/3/tutorial/introduction.html)
# - [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
# - [Understanding Python Variables and Memory](https://realpython.com/python-variables/)
