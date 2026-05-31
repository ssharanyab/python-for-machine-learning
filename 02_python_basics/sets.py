# %% [markdown]
# # Sets
#
# ## Why this matters
# A Set is an unordered collection of unique elements. In Machine Learning, sets are heavily used to find unique categories in a dataset, perform fast membership testing (e.g., checking if a word is in a list of stopwords), and handle set operations like unions and intersections across data subsets.
#
# ## Learning Objectives
# - Learn how to define and modify sets.
# - Understand the property of uniqueness in sets.
# - Perform mathematical set operations: union, intersection, difference, symmetric difference.
# - Understand the time-complexity benefits of sets for membership testing compared to lists.
#
# ## Concept Explanation
# Sets are written with curly brackets `{}`, or created using the `set()` constructor. Sets cannot have two items with the same value. They are unordered, unindexed, and unchangeable (though you can add or remove items).
#
# ## Beginner Examples

# %%
# Example 1: Creating a set and uniqueness
my_set = {1, 2, 2, 3, 4, 4, 4}
print(f"Unique elements: {my_set}")

# %%
# Example 2: Adding and removing elements
fruits = {"apple", "banana"}
fruits.add("cherry")
fruits.remove("banana") # raises KeyError if not found
fruits.discard("orange") # does NOT raise error if not found
print(f"Updated set: {fruits}")

# %%
# Example 3: Fast Membership Testing
# Checking 'in' on a set is O(1) time complexity
stopwords = {"the", "a", "an", "is"}
word = "the"
print(f"Is '{word}' a stopword? {word in stopwords}")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: Union and Intersection
set_A = {1, 2, 3, 4}
set_B = {3, 4, 5, 6}

# Union (All elements from both)
print(f"Union: {set_A | set_B}")

# Intersection (Elements present in both)
print(f"Intersection: {set_A & set_B}")

# %%
# Example 5: Difference and Symmetric Difference
# Difference (Elements in A but not in B)
print(f"Difference (A - B): {set_A - set_B}")

# Symmetric Difference (Elements in either A or B, but not both)
print(f"Symmetric Difference: {set_A ^ set_B}")

# %%
# Example 6: Subsets and Supersets
small_set = {1, 2}
large_set = {1, 2, 3, 4}
print(f"Is small_set a subset of large_set? {small_set.issubset(large_set)}")
print(f"Is large_set a superset of small_set? {large_set.issuperset(small_set)}")

# %% [markdown]
# ## Machine Learning Relevance
# Sets are used when preparing vocabulary for Natural Language Processing (NLP), or checking overlapping features between different datasets (e.g., train and test sets).

# %%
# ML Example: Finding Unique Vocabulary
document = "machine learning is fun and machine learning is powerful"
words = document.split()
# Find the unique words (vocabulary)
vocabulary = set(words)
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Vocabulary: {vocabulary}")

# %% [markdown]
# ## Common Mistakes
# 1. **Creating an Empty Set:** Using `{}` creates an empty dictionary, not an empty set. Use `set()` for an empty set.
# 2. **Assuming Order:** Sets are unordered. Do not write code that assumes items will be returned or printed in the order they were added.
# 3. **Un-hashable Types:** Sets cannot contain mutable elements like lists or dictionaries.

# %% [markdown]
# ## Interview Questions
# 1. What is the difference between a list and a set in Python?
# 2. How do you create an empty set?
# 3. Explain the time complexity of checking if an item exists in a set vs. a list.
# 4. What is the difference between the `.remove()` and `.discard()` methods?
# 5. Why can't you put a list inside a set?

# %% [markdown]
# ## Practice Problems
# 1. Given a list with duplicates `[1, 2, 2, 3, 4, 4, 5]`, use a set to create a list of unique items.
# 2. Find the common elements between `list_a = [1, 2, 3, 4]` and `list_b = [3, 4, 5, 6]`.
# 3. Given two sets of features, `model_a_features` and `model_b_features`, find the features that are unique to each model (not shared).
# 4. Write code to check if two sets are disjoint (have no elements in common).
# 5. Create an empty set, add the numbers 1, 2, 3, and then remove 2 safely.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: Unique items
duplicates = [1, 2, 2, 3, 4, 4, 5]
unique_list = list(set(duplicates))
print(unique_list)

# %%
# Solution 2: Common elements
list_a = [1, 2, 3, 4]
list_b = [3, 4, 5, 6]
common = list(set(list_a) & set(list_b))
print(common)

# %%
# Solution 3: Unique to each
model_a = {"age", "income", "height"}
model_b = {"age", "income", "weight"}
unique_features = model_a ^ model_b
print(unique_features)

# %%
# Solution 4: Disjoint sets
set1 = {1, 2}
set2 = {3, 4}
print(set1.isdisjoint(set2))

# %%
# Solution 5: Empty set operations
empty = set()
empty.update([1, 2, 3])
empty.discard(2)
print(empty)

# %% [markdown]
# ## Further Reading
# - Python Data Structures: Sets
# - TimeComplexity in Python
