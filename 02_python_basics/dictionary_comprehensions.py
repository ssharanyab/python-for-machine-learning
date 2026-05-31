# %% [markdown]
# # Dictionary Comprehensions
#
# ## Why this matters
# Similar to list comprehensions, dictionary comprehensions provide a quick, readable way to construct dictionaries. In Machine Learning, you frequently manage hyperparameters, map category labels to integers (encoding), or aggregate model results. Dictionary comprehensions excel in these scenarios.
#
# ## Learning Objectives
# - Understand the syntax of dictionary comprehensions.
# - Learn how to map keys to values efficiently.
# - Use conditionals to filter dictionary creation.
# - Apply dictionary comprehensions to invert dictionaries or process key-value pairs.
#
# ## Concept Explanation
# A dictionary comprehension creates a dictionary using a concise syntax.
# Syntax: `{key_expression: value_expression for item in iterable if condition}`
# You can also iterate over existing dictionaries using `.items()`.
#
# ## Beginner Examples

# %%
# Example 1: Basic Dictionary Comprehension
# Create a dictionary of squares
squares_dict = {x: x**2 for x in range(5)}
print(f"Squares Dict: {squares_dict}")

# %%
# Example 2: Mapping a list of strings to their lengths
words = ["data", "science", "machine", "learning"]
word_lengths = {word: len(word) for word in words}
print(f"Word Lengths: {word_lengths}")

# %%
# Example 3: Filtering items
# Only include squares for even numbers
even_squares_dict = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"Even Squares: {even_squares_dict}")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: Inverting a Dictionary
# Swap keys and values
original = {'a': 1, 'b': 2, 'c': 3}
inverted = {value: key for key, value in original.items()}
print(f"Inverted: {inverted}")

# %%
# Example 5: If-Else in Dictionary Values
# Assign categories based on value
scores = {'Alice': 85, 'Bob': 50, 'Charlie': 95}
grades = {name: ('Pass' if score >= 60 else 'Fail') for name, score in scores.items()}
print(f"Grades: {grades}")

# %%
# Example 6: Combining two lists into a dictionary
keys = ['accuracy', 'precision', 'recall']
values = [0.95, 0.92, 0.90]
metrics = {k: v for k, v in zip(keys, values)}
print(f"Metrics: {metrics}")

# %% [markdown]
# ## Machine Learning Relevance
# Dictionary comprehensions are highly useful for encoding categorical variables, standardizing configurations, or mapping dataset indices to their actual labels.

# %%
# ML Example: Label Encoding
classes = ["cat", "dog", "bird"]
# Create a mapping from class name to an integer index
label_encoder = {class_name: idx for idx, class_name in enumerate(classes)}
print(f"Label Encoder: {label_encoder}")

# Reverse mapping
index_to_class = {idx: class_name for class_name, idx in label_encoder.items()}
print(f"Decoder: {index_to_class}")

# %% [markdown]
# ## Common Mistakes
# 1. **Key Collisions:** When inverting a dictionary, if the original values are not unique, keys in the new dictionary will overwrite each other.
# 2. **Complex Logic:** Just like list comprehensions, packing too much logic into a dictionary comprehension reduces readability.
# 3. **Forgetting `.items()`:** Trying to iterate over a dictionary directly (`for k in dict`) only yields keys. You must use `.items()` to get both keys and values.

# %% [markdown]
# ## Interview Questions
# 1. What is a dictionary comprehension and what is its syntax?
# 2. How do you swap the keys and values of a dictionary using a comprehension?
# 3. What happens if you try to invert a dictionary that has duplicate values using a dictionary comprehension?
# 4. How can you use `zip()` within a dictionary comprehension?
# 5. Provide an example of conditional logic applied to the values of a dictionary comprehension.

# %% [markdown]
# ## Practice Problems
# 1. Given a list of temperatures in Celsius `[0, 10, 20, 30]`, create a dictionary mapping the Celsius temperature to its Fahrenheit equivalent.
# 2. Given a dictionary of items and their prices `{'apple': 2, 'banana': 1, 'steak': 15}`, create a new dictionary with a 10% discount applied to the prices.
# 3. Create a dictionary that maps numbers from 1 to 10 to True if they are even, and False if they are odd.
# 4. Given a string `text = "hello world"`, create a dictionary mapping each character to its frequency count.
# 5. Filter a dictionary of hyperparameters `{'lr': 0.01, 'batch': 32, 'momentum': 0.9}` to only include keys that start with the letter 'm' or 'l'.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Celsius to Fahrenheit
celsius = [0, 10, 20, 30]
f_dict = {c: (c * 9/5) + 32 for c in celsius}
print(f_dict)

# %%
# Solution 2: Apply discount
prices = {'apple': 2, 'banana': 1, 'steak': 15}
discounted = {item: price * 0.9 for item, price in prices.items()}
print(discounted)

# %%
# Solution 3: Even/Odd mapping
even_odd = {x: (True if x % 2 == 0 else False) for x in range(1, 11)}
print(even_odd)

# %%
# Solution 4: Character frequency
text = "hello world"
freq = {char: text.count(char) for char in set(text)}
print(freq)

# %%
# Solution 5: Filter keys
hyperparams = {'lr': 0.01, 'batch': 32, 'momentum': 0.9}
filtered = {k: v for k, v in hyperparams.items() if k.startswith('m') or k.startswith('l')}
print(filtered)

# %% [markdown]
# ## Further Reading
# - PEP 274 – Dict Comprehensions
# - Python Data Structures Documentation
