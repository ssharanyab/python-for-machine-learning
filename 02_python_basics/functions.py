# %% [markdown]
# # Functions
#
# ## Why this matters
# Functions are the building blocks of reusable code in Python. In machine learning, you will often need to perform the same data preprocessing, model evaluation, or training steps repeatedly. Wrapping these steps in functions makes your code modular, readable, and less prone to errors.
#
# ## Learning Objectives
# - Understand how to define and call functions in Python.
# - Learn about positional and keyword arguments.
# - Understand default parameters and arbitrary arguments (*args, **kwargs).
# - Understand variable scope (local vs. global).
# - Learn how to return single and multiple values from functions.
#
# ## Concept Explanation
# A function is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing. In Python, a function is defined using the `def` keyword, followed by the function name and parentheses `()`.
#
# ## Beginner Examples

# %%
# Example 1: Basic Function Definition and Call
def greet(name):
    """Prints a simple greeting."""
    print(f"Hello, {name}!")

greet("Alice")

# %%
# Example 2: Return Values
def add_numbers(a, b):
    """Returns the sum of two numbers."""
    return a + b

result = add_numbers(5, 3)
print(f"Sum: {result}")

# %%
# Example 3: Default Arguments
def power(base, exponent=2):
    """Calculates power with a default exponent of 2."""
    return base ** exponent

print(f"Square of 4: {power(4)}")
print(f"Cube of 4: {power(4, 3)}")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: *args (Arbitrary Positional Arguments)
def calculate_mean(*args):
    """Calculates the mean of an arbitrary number of arguments."""
    if len(args) == 0:
        return 0
    return sum(args) / len(args)

print(f"Mean of 1, 2, 3: {calculate_mean(1, 2, 3)}")
print(f"Mean of 10, 20, 30, 40: {calculate_mean(10, 20, 30, 40)}")

# %%
# Example 5: **kwargs (Arbitrary Keyword Arguments)
def print_model_hyperparameters(**kwargs):
    """Prints hyperparameter key-value pairs."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_model_hyperparameters(learning_rate=0.01, batch_size=32, epochs=100)

# %%
# Example 6: Returning Multiple Values
def summarize_data(data):
    """Returns the min, max, and sum of a list."""
    return min(data), max(data), sum(data)

min_val, max_val, total = summarize_data([1, 5, 2, 8, 3])
print(f"Min: {min_val}, Max: {max_val}, Total: {total}")

# %% [markdown]
# ## Machine Learning Relevance
# In machine learning, functions are used heavily to encapsulate logic. For example:
# - Custom data loading and cleaning functions.
# - Feature engineering transformations.
# - Defining custom loss functions or evaluation metrics.
# - Wrapping the training loop of a neural network.

# %%
# ML Example: A basic normalization function
def min_max_normalize(features):
    """Applies Min-Max normalization to a list of features."""
    min_f = min(features)
    max_f = max(features)
    range_f = max_f - min_f
    if range_f == 0:
        return [0 for _ in features]
    return [(f - min_f) / range_f for f in features]

raw_data = [10, 20, 30, 40, 50]
normalized_data = min_max_normalize(raw_data)
print(f"Normalized Data: {normalized_data}")

# %% [markdown]
# ## Common Mistakes
# 1. **Mutable Default Arguments:** Using a mutable object (like a list or dict) as a default argument. It retains its state between calls.
# 2. **Shadowing Variables:** Using the same name for a local variable and a global variable, causing confusion about scope.
# 3. **Missing Return Statements:** Forgetting to return a value, causing the function to return `None`.

# %% [markdown]
# ## Interview Questions
# 1. What is the difference between arguments and parameters?
# 2. How do `*args` and `**kwargs` work in Python? Provide an example.
# 3. Explain the difference between local and global scope.
# 4. Why is it dangerous to use a mutable object as a default argument in a function?
# 5. Can a function return multiple values in Python? If so, how?

# %% [markdown]
# ## Practice Problems
# 1. Write a function `calculate_accuracy(predictions, labels)` that takes two lists of the same length and returns the percentage of matching elements.
# 2. Write a function `standardize(data)` that takes a list of numbers and returns a new list where each number is standardized (z-score: (x - mean) / std_dev). You can use simple mathematical approximations.
# 3. Write a function `create_pipeline(*functions)` that takes an arbitrary number of functions and returns a new function that applies them sequentially to an input.
# 4. Write a function `get_stats(data)` that returns a dictionary containing the 'mean', 'min', and 'max' of the input list.
# 5. Fix the following function so that the default argument doesn't carry over state:
#    `def add_item(item, lst=[]): lst.append(item); return lst`

# %% [markdown]
# ## Solutions

# %%
# Solution 1: calculate_accuracy
def calculate_accuracy(predictions, labels):
    if len(predictions) != len(labels) or len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)

print(calculate_accuracy([1, 0, 1, 1], [1, 0, 0, 1]))

# %%
# Solution 2: standardize
def standardize(data):
    n = len(data)
    if n == 0: return []
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    if std_dev == 0: return [0.0 for _ in data]
    return [(x - mean) / std_dev for x in data]

print(standardize([1, 2, 3, 4, 5]))

# %%
# Solution 3: create_pipeline
def create_pipeline(*functions):
    def pipeline(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return pipeline

def double(x): return x * 2
def add_one(x): return x + 1

my_pipeline = create_pipeline(double, add_one)
print(my_pipeline(3)) # (3 * 2) + 1 = 7

# %%
# Solution 4: get_stats
def get_stats(data):
    if not data: return {}
    return {
        'mean': sum(data) / len(data),
        'min': min(data),
        'max': max(data)
    }

print(get_stats([10, 20, 30]))

# %%
# Solution 5: Fix mutable default argument
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(add_item(1))
print(add_item(2))

# %% [markdown]
# ## Further Reading
# - Python Official Documentation on Functions
# - PEP 8 Style Guide for Python Code (Function Naming)
