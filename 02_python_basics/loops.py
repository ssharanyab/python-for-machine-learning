# %% [markdown]
# # Loops and Iterations
#
# # Why this matters
# Machine learning models learn iteratively. Training neural networks involves looping over datasets multiple times (epochs) and stepping through subsets of data (batches). Without loops, modern iterative optimization (like Gradient Descent) would be impossible to code.
#
# # Learning Objectives
# 1. Master the `for` loop to iterate over collections (lists, tuples, dicts).
# 2. Understand the `while` loop for condition-based iteration.
# 3. Learn loop control statements: `break`, `continue`, and `pass`.
# 4. Use `range()`, `enumerate()`, and `zip()` built-in functions effectively.
#
# # Concept Explanation
# - **`for` loop:** Iterates over items of any sequence (e.g., a list or a string) in the order they appear.
# - **`while` loop:** Repeatedly executes a target statement as long as a given condition is True.
# - **`break`:** Terminates the loop entirely.
# - **`continue`:** Skips the rest of the current iteration and moves to the next one.
#
# # Beginner Examples
# Exploring basic loops with lists and ranges.

# %%
# Basic For loop
architectures = ["CNN", "RNN", "Transformer"]
for arch in architectures:
    print(f"Training a {arch} model...")

# Using range() for numerical iteration (e.g., Epochs)
epochs = 3
for i in range(epochs):
    print(f"Epoch {i+1}/{epochs} completed.")

# Basic While loop
tolerance = 0.01
error = 0.5
steps = 0

while error > tolerance:
    error *= 0.5  # Simulate error decreasing
    steps += 1

print(f"Converged after {steps} steps with error {error:.4f}")

# %% [markdown]
# # Intermediate Examples
# Iterating over dictionaries, handling complex data flows with `enumerate` and `zip`, and using loop controls.

# %%
# Enumerate: getting both index and value (Useful for batch indices)
batches = [100, 200, 150]
for idx, batch_size in enumerate(batches):
    print(f"Batch {idx} processed {batch_size} items.")

# Zip: iterating over two lists in parallel
features = ["size", "bedrooms", "location"]
weights = [50.5, 10.2, 5.0]

print("\nModel Weights:")
for feature, weight in zip(features, weights):
    print(f"- {feature}: {weight}")

# Loop Controls: early stopping using break and continue
losses = [1.5, 1.2, "NaN", 0.9, 0.8]
for val in losses:
    if val == "NaN":
        print("Warning: Loss is NaN, skipping iteration.")
        continue # Skip this step
    
    if val < 1.0:
        print(f"Target loss {val} achieved. Stopping training.")
        break # Exit loop completely
    print(f"Loss: {val}")

# %% [markdown]
# # Machine Learning Relevance
# The training loop in PyTorch or TensorFlow is structurally a massive `for` loop. You iterate `for epoch in range(epochs):` and inside that, `for batch in dataloader:`. You use `break` for early stopping, `continue` to skip corrupted data files, and `zip` to iterate over inputs and targets simultaneously.
#
# # Common Mistakes
# 1. Creating infinite `while` loops by forgetting to update the condition variable inside the loop.
# 2. Modifying a list while iterating over it, causing skipped elements or infinite loops. (Iterate over a copy instead: `for item in my_list.copy():`).
# 3. Not using `enumerate`, resulting in un-Pythonic code like `for i in range(len(my_list)): print(my_list[i])`.
# 4. Using inefficient nested loops instead of vectorized operations via NumPy/Pandas. (Loops are slow in pure Python!).
#
# # Interview Questions
# 1. **What is the difference between `break` and `continue`?**
#    *Answer:* `break` exits the loop entirely. `continue` skips the remaining code in the current iteration and jumps to the next iteration.
# 2. **What does the `enumerate()` function do?**
#    *Answer:* It adds a counter to an iterable and returns it as an enumerate object, yielding tuples of `(index, value)`.
# 3. **How does a `for-else` block work in Python?**
#    *Answer:* The `else` block executes after the loop completes normally. If the loop is terminated by a `break` statement, the `else` block is skipped.
# 4. **Why are `for` loops in pure Python considered slow for large numerical datasets?**
#    *Answer:* Because Python is interpreted and dynamically typed, there is significant overhead in type checking and dispatching on every iteration. Vectorized C/C++ implementations (like NumPy) bypass this.
# 5. **What is `zip()` and what happens if the iterables are of different lengths?**
#    *Answer:* `zip()` aggregates elements from iterables. It stops when the shortest iterable is exhausted. (Python 3.10 added `strict=True` to raise an error if lengths differ).
#
# # Practice Problems
# 1. Write a `for` loop that prints the squares of numbers from 1 to 5.
# 2. Given a list `data = [10, -5, 20, -1, 30]`, write a loop that adds positive numbers to a `clean_data` list and `continue`s when it sees a negative number.
# 3. Iterate over the dictionary `hyperparams = {"lr": 0.01, "batch": 32, "epochs": 100}` and print strings like "lr is set to 0.01".
# 4. Use `zip()` to combine `preds = [0, 1, 1]` and `labels = [0, 1, 0]`. Print whether each prediction is correct.
# 5. Write a `while` loop that halves a learning rate starting from `0.1` until it drops below `0.001`. Print the learning rate at each step.
#
# # Solutions

# %%
# Solution to Problem 1
for i in range(1, 6):
    print(i**2)

# %%
# Solution to Problem 2
data = [10, -5, 20, -1, 30]
clean_data = []
for num in data:
    if num < 0:
        continue
    clean_data.append(num)
print("Cleaned data:", clean_data)

# %%
# Solution to Problem 3
hyperparams = {"lr": 0.01, "batch": 32, "epochs": 100}
for key, value in hyperparams.items():
    print(f"{key} is set to {value}")

# %%
# Solution to Problem 4
preds = [0, 1, 1]
labels = [0, 1, 0]
for p, l in zip(preds, labels):
    is_correct = p == l
    print(f"Pred: {p}, Label: {l} -> Correct: {is_correct}")

# %%
# Solution to Problem 5
lr = 0.1
while lr >= 0.001:
    print(f"Current LR: {lr:.5f}")
    lr /= 2

# %% [markdown]
# # Further Reading
# - [Python Loops](https://wiki.python.org/moin/ForLoop)
# - [Itertools - Functions creating iterators for efficient looping](https://docs.python.org/3/library/itertools.html)
