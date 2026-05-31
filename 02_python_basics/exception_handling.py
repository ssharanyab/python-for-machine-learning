# %% [markdown]
# # Exception Handling
#
# ## Why this matters
# In machine learning, data pipelines often break due to messy or unexpected data (missing files, malformed strings, division by zero). Exception handling allows your code to gracefully catch these errors, log them, and continue processing rather than crashing the entire script.
#
# ## Learning Objectives
# - Understand the `try`, `except`, `else`, and `finally` blocks.
# - Learn to catch specific exceptions rather than using bare except clauses.
# - Raise custom exceptions.
# - Use exceptions effectively in data processing tasks.
#
# ## Concept Explanation
# Exceptions are errors that occur during execution. You can handle these errors using a `try...except` block.
# - `try`: The block of code to be tested for errors.
# - `except`: The block of code to execute if an error occurs.
# - `else`: The block of code to execute if no errors were raised.
# - `finally`: The block of code that always executes, regardless of errors.
#
# ## Beginner Examples

# %%
# Example 1: Basic Try-Except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Cannot divide by zero!")

# %%
# Example 2: Catching specific exceptions
try:
    num = int("hello")
except ValueError:
    print("Error: Invalid integer conversion.")

# %%
# Example 3: Using Else and Finally
try:
    value = int("42")
except ValueError:
    print("Conversion failed.")
else:
    print(f"Conversion successful: {value}")
finally:
    print("Execution complete.")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: Catching Multiple Exceptions
def process_data(data, index):
    try:
        val = 100 / data[index]
        return val
    except IndexError:
        print("Error: Index out of bounds.")
    except ZeroDivisionError:
        print("Error: Data value is zero.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

process_data([1, 0, 3], 1)
process_data([1, 0, 3], 5)

# %%
# Example 5: Raising Exceptions
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative.")
    return age

try:
    check_age(-5)
except ValueError as e:
    print(f"Caught an error: {e}")

# %% [markdown]
# ## Machine Learning Relevance
# Exception handling is crucial when iterating over large datasets or parsing JSON files, where a single bad record shouldn't stop the whole training or inference loop.

# %%
# ML Example: Robust Feature Scaling
def safe_normalize(data):
    try:
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        if range_val == 0:
            raise ValueError("Data has zero variance; cannot normalize.")
        return [(x - min_val) / range_val for x in data]
    except ValueError as e:
        print(f"Normalization skipped: {e}")
        return data
    except TypeError:
        print("Error: Data must contain numbers.")
        return []

print(safe_normalize([5, 5, 5]))
print(safe_normalize(["a", "b"]))

# %% [markdown]
# ## Common Mistakes
# 1. **Bare Excepts:** Using just `except:` without specifying the error type. This catches `KeyboardInterrupt` and `SystemExit`, making it hard to stop the script. Always catch specific exceptions or `Exception`.
# 2. **Silent Failures:** Catching an exception using `pass` and not logging it, making debugging extremely difficult.
# 3. **Overusing Exceptions:** Using exceptions for normal control flow (like checking if a file exists) when simple `if` conditions would be faster and cleaner.

# %% [markdown]
# ## Interview Questions
# 1. Explain the purpose of `try`, `except`, `else`, and `finally`.
# 2. Why is it considered a bad practice to use a bare `except:` block?
# 3. How do you raise a custom exception in Python?
# 4. What is the difference between `ValueError` and `TypeError`?
# 5. In a data processing pipeline, how would you handle a missing column error without stopping the script?

# %% [markdown]
# ## Practice Problems
# 1. Write a function that safely converts a string to a float, returning 0.0 if it fails.
# 2. Create a script that tries to open a file `missing.txt` for reading and gracefully handles the `FileNotFoundError`.
# 3. Write a function that accepts a dictionary and a key. Try to access the key. If a `KeyError` occurs, return a default string "Not Found".
# 4. Define a custom exception class `NegativeLossError`. Write a function that calculates loss and raises this error if the loss is negative.
# 5. Write a try-except block that catches multiple exceptions (`TypeError` and `ValueError`) in a single `except` line.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Safe Float
def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

print(safe_float("3.14"))
print(safe_float("abc"))

# %%
# Solution 2: File Not Found
try:
    with open("missing.txt", "r") as f:
        data = f.read()
except FileNotFoundError:
    print("The file was not found.")

# %%
# Solution 3: Key Access
def get_val(d, key):
    try:
        return d[key]
    except KeyError:
        return "Not Found"

print(get_val({'a': 1}, 'b'))

# %%
# Solution 4: Custom Exception
class NegativeLossError(Exception):
    pass

def check_loss(loss):
    if loss < 0:
        raise NegativeLossError("Loss cannot be less than 0.")
    return True

try:
    check_loss(-0.5)
except NegativeLossError as e:
    print(e)

# %%
# Solution 5: Multiple Exceptions
def process(val):
    try:
        num = int(val)
        return 100 / num
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"Error processed: {type(e).__name__}")

process("a")
process(0)

# %% [markdown]
# ## Further Reading
# - Python Errors and Exceptions
# - Real Python: Python Exceptions
