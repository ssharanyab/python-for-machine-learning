# %% [markdown]
# # Lambda Functions
#
# ## Why this matters
# Lambda functions are small, anonymous functions. In Machine Learning, especially when working with data manipulation libraries like Pandas, you will frequently need to apply quick, inline transformations to data. Lambda functions provide a concise syntax for this without needing a full `def` block.
#
# ## Learning Objectives
# - Understand what lambda functions are and their syntax.
# - Learn how to use lambda functions with built-in functions like `map()`, `filter()`, and `sorted()`.
# - Understand the limitations of lambda functions compared to regular functions.
# - Learn how lambda functions are applied in Pandas for data transformations.
#
# ## Concept Explanation
# A lambda function is a small anonymous function defined using the `lambda` keyword. It can take any number of arguments but can only have one expression. The expression is evaluated and returned.
# Syntax: `lambda arguments: expression`
#
# ## Beginner Examples

# %%
# Example 1: Basic Lambda Function
square = lambda x: x ** 2
print(f"Square of 5: {square(5)}")

# %%
# Example 2: Multiple Arguments
add = lambda x, y: x + y
print(f"Sum of 3 and 7: {add(3, 7)}")

# %%
# Example 3: Using with sorted()
# Sorting a list of tuples by the second element
points = [(1, 5), (3, 2), (2, 8)]
points_sorted = sorted(points, key=lambda point: point[1])
print(f"Sorted points: {points_sorted}")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: Using with map()
# Applying a transformation to all elements in a list
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x**2, numbers))
print(f"Squared numbers: {squared_numbers}")

# %%
# Example 5: Using with filter()
# Filtering elements based on a condition
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Even numbers: {even_numbers}")

# %%
# Example 6: Conditional expressions inside a lambda
# Classify numbers as positive or negative
classify = lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Zero"
print(f"10 is {classify(10)}, -5 is {classify(-5)}, 0 is {classify(0)}")

# %% [markdown]
# ## Machine Learning Relevance
# In data preprocessing (e.g., with Pandas DataFrames), `apply()` is often used with lambda functions to quickly engineer new features or clean existing ones.

# %%
# ML Example: Simulating Pandas apply with a dictionary list
data = [{'age': 25, 'income': 50000}, {'age': 40, 'income': 80000}, {'age': 22, 'income': 30000}]
# Add a new 'seniority' feature based on age
list(map(lambda row: row.update({'seniority': 'Senior' if row['age'] > 30 else 'Junior'}), data))
print(f"Transformed data: {data}")

# %% [markdown]
# ## Common Mistakes
# 1. **Overusing Lambda Functions:** Making them too complex. If a lambda function has too much logic, it becomes hard to read, and a standard `def` function should be used instead.
# 2. **Assigning to Variables:** PEP 8 discourages assigning lambda expressions to variables (`f = lambda x: x`). Instead, use `def f(x): return x`.
# 3. **Misunderstanding Scope:** Lambda functions use lexical scoping, capturing variables from the surrounding environment.

# %% [markdown]
# ## Interview Questions
# 1. What is a lambda function in Python and how does it differ from a regular function?
# 2. Can a lambda function contain multiple expressions or statements?
# 3. When would you choose to use a lambda function over a regular function?
# 4. How do you use a lambda function to sort a list of dictionaries by a specific key?
# 5. Explain how `map()` and `filter()` work with lambda functions.

# %% [markdown]
# ## Practice Problems
# 1. Write a lambda function that checks if a string is a palindrome.
# 2. Given a list of strings, use `filter()` and a lambda function to keep only the strings that have more than 5 characters.
# 3. Given a list of numbers, use `map()` and a lambda function to convert them from Celsius to Fahrenheit.
# 4. Sort a list of tuples representing `(name, age)` by age in descending order using a lambda function.
# 5. Create a lambda function that takes two arguments and returns their product if both are positive, otherwise returns 0.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Palindrome checker
is_palindrome = lambda s: s == s[::-1]
print(is_palindrome("racecar"))
print(is_palindrome("hello"))

# %%
# Solution 2: Filter strings
words = ["apple", "cat", "banana", "dog", "elephant"]
long_words = list(filter(lambda w: len(w) > 5, words))
print(long_words)

# %%
# Solution 3: Celsius to Fahrenheit
celsius = [0, 20, 37, 100]
fahrenheit = list(map(lambda c: (c * 9/5) + 32, celsius))
print(fahrenheit)

# %%
# Solution 4: Sort tuples by age descending
people = [("Alice", 25), ("Bob", 30), ("Charlie", 22)]
sorted_people = sorted(people, key=lambda p: p[1], reverse=True)
print(sorted_people)

# %%
# Solution 5: Product if positive
conditional_product = lambda x, y: x * y if x > 0 and y > 0 else 0
print(conditional_product(5, 4))
print(conditional_product(-2, 4))

# %% [markdown]
# ## Further Reading
# - Python Documentation on Lambda Expressions
# - Real Python: How to Use Python Lambda Functions
