# %% [markdown]
# # List Comprehensions
#
# ## Why this matters
# List comprehensions provide a concise, readable, and highly optimized way to create lists in Python. In data science and machine learning, you will often need to iterate over datasets to transform features, filter outliers, or compute metrics. List comprehensions allow you to do this in a single line, making your code cleaner and faster.
#
# ## Learning Objectives
# - Understand the syntax of list comprehensions.
# - Learn how to replace `for` loops with list comprehensions.
# - Use conditionals (`if`/`else`) within list comprehensions.
# - Work with nested list comprehensions for multi-dimensional data.
# - Understand the performance benefits over standard loops.
#
# ## Concept Explanation
# A list comprehension consists of brackets containing an expression followed by a `for` clause, then zero or more `for` or `if` clauses. The expressions can be anything, meaning you can put in all kinds of objects in lists.
# Syntax: `[expression for item in iterable if condition]`
#
# ## Beginner Examples

# %%
# Example 1: Basic List Comprehension
# Create a list of squares
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# %%
# Example 2: Filtering with 'if'
# Get only even numbers
evens = [x for x in range(10) if x % 2 == 0]
print(f"Evens: {evens}")

# %%
# Example 3: String manipulation
# Convert a list of strings to uppercase
words = ["hello", "world", "ml"]
upper_words = [word.upper() for word in words]
print(f"Uppercase: {upper_words}")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: If-Else in List Comprehension
# Note: The syntax changes when using if-else. It comes BEFORE the 'for' loop.
# Label numbers as Even or Odd
labels = ["Even" if x % 2 == 0 else "Odd" for x in range(5)]
print(f"Labels: {labels}")

# %%
# Example 5: Extracting features from a list of dictionaries
data = [{'value': 10}, {'value': 20}, {'value': 30}]
values = [item['value'] for item in data]
print(f"Extracted Values: {values}")

# %%
# Example 6: Nested List Comprehensions
# Flatten a 2D matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(f"Flattened Matrix: {flattened}")

# %% [markdown]
# ## Machine Learning Relevance
# List comprehensions are highly relevant when preprocessing text data (e.g., tokenization, lowercasing, removing stopwords) or computing batch-wise loss calculations.

# %%
# ML Example: Text Preprocessing
stopwords = ["is", "the", "in", "and", "a"]
text = "The machine learning algorithm is in a new phase"
# Tokenize, lowercase, and remove stopwords
clean_tokens = [word.lower() for word in text.split() if word.lower() not in stopwords]
print(f"Cleaned Tokens: {clean_tokens}")

# %% [markdown]
# ## Common Mistakes
# 1. **Over-complexity:** Making a list comprehension too long or deeply nested. If it spans multiple lines and is hard to read, use a standard `for` loop.
# 2. **Memory Exhaustion:** List comprehensions create the entire list in memory. For massive datasets, consider generator expressions instead (using parentheses `()` instead of brackets `[]`).
# 3. **Syntax Errors with If-Else:** Forgetting that `if` alone goes at the end, while `if-else` goes before the `for`.

# %% [markdown]
# ## Interview Questions
# 1. What is a list comprehension and why is it useful?
# 2. How do you incorporate an `if-else` statement within a list comprehension?
# 3. What is the difference between a list comprehension and a generator expression?
# 4. How would you flatten a list of lists using a list comprehension?
# 5. Are list comprehensions always faster than `for` loops? Explain.

# %% [markdown]
# ## Practice Problems
# 1. Given a list of numbers, create a new list containing only the positive numbers.
# 2. Given a list of words, create a list of their lengths.
# 3. Create a list of tuples `(number, cube)` for numbers from 1 to 5.
# 4. Given two lists, `A = [1, 2, 3]` and `B = [a, b, c]`, create a Cartesian product list of tuples: `[(1, 'a'), (1, 'b'), ...]`.
# 5. Write a list comprehension that extracts the diagonal elements of a 3x3 matrix represented as a list of lists.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Positive numbers
nums = [-5, 3, -1, 10, 0, -2]
positives = [x for x in nums if x > 0]
print(positives)

# %%
# Solution 2: Word lengths
words_list = ["apple", "banana", "cherry"]
lengths = [len(w) for w in words_list]
print(lengths)

# %%
# Solution 3: Cubes
cubes = [(x, x**3) for x in range(1, 6)]
print(cubes)

# %%
# Solution 4: Cartesian product
A = [1, 2]
B = ['a', 'b']
product = [(a, b) for a in A for b in B]
print(product)

# %%
# Solution 5: Diagonal elements
mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
diagonal = [mat[i][i] for i in range(len(mat))]
print(diagonal)

# %% [markdown]
# ## Further Reading
# - Python Tutorial: Data Structures
# - Real Python: When to Use a List Comprehension in Python
