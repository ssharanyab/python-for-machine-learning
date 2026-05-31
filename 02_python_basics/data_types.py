# %% [markdown]
# # Data Types
#
# # Why this matters
# Data is the lifeblood of Machine Learning. Knowing how Python represents different types of data—integers, floats, strings, booleans, lists, tuples, sets, and dictionaries—is essential for feature engineering, data cleaning, and model preparation.
#
# # Learning Objectives
# 1. Identify and use Python's built-in primitive data types (int, float, bool, str).
# 2. Work with collection data types (list, tuple, dict, set).
# 3. Understand mutability vs. immutability.
# 4. Perform type casting (conversion between types).
#
# # Concept Explanation
# Python has several core data types:
# - **Numeric Types:** `int` (integers), `float` (decimals), `complex` (complex numbers).
# - **Sequence Types:** `str` (strings), `list` (ordered, mutable sequences), `tuple` (ordered, immutable sequences).
# - **Mapping Type:** `dict` (key-value pairs).
# - **Set Types:** `set` (unordered collections of unique items).
# - **Boolean Type:** `bool` (`True` or `False`).
# An object's mutability determines if it can be changed after creation. Lists and dicts are mutable; strings and tuples are immutable.
#
# # Beginner Examples
# Let's explore basic primitives and sequences.

# %%
# Primitives
integer_val = 42            # int
float_val = 3.14159         # float
boolean_val = True          # bool
string_val = "Neural Net"   # str

print(type(integer_val), type(float_val), type(boolean_val), type(string_val))

# Collections
features_list = ["age", "height", "weight"]      # list (mutable)
hyperparams_tuple = (0.01, 32, 100)              # tuple (immutable)
unique_labels = {"cat", "dog", "bird"}           # set (unordered, unique)
model_config = {"optimizer": "adam", "lr": 0.01} # dict (key-value)

print(type(features_list), type(hyperparams_tuple), type(unique_labels), type(model_config))

# %% [markdown]
# # Intermediate Examples
# Here we will look at type casting, dictionary manipulation, and mutability.

# %%
# Type Casting
str_num = "100"
actual_num = int(str_num)
float_num = float(str_num)
print(f"Casting: String '{str_num}' -> Int {actual_num} -> Float {float_num}")

# Mutability test
features_list[0] = "age_years" # Allowed!
print("Modified list:", features_list)

try:
    hyperparams_tuple[0] = 0.05 # Not allowed! Raises TypeError
except TypeError as e:
    print(f"Tuple immutability error: {e}")

# Dictionary manipulation
model_config["epochs"] = 50 # Add new key-value pair
model_config["lr"] = 0.005  # Update existing
print("Updated Dictionary:", model_config)

# %% [markdown]
# # Machine Learning Relevance
# Machine Learning models require numerical input. Text data (`str`) must be encoded into numbers (`int` or `float`). Categorical labels might be stored in `set`s to find unique classes, and hyperparameters are almost always passed around as `dict`s. Understanding tuples is important because matrix shapes in NumPy/Pandas/TensorFlow are returned as tuples (e.g., `(1000, 4)` for a dataset with 1000 rows and 4 columns).
#
# # Common Mistakes
# 1. Trying to append to a tuple, causing a TypeError.
# 2. Using mutable default arguments in functions (e.g., `def train(config={})`), which retains state across calls.
# 3. Forgetting that dictionary keys must be immutable (you can use a tuple as a key, but not a list).
# 4. Confusing sets and dictionaries since both use curly braces `{}`. `{}` creates an empty dict, `set()` creates an empty set.
#
# # Interview Questions
# 1. **What is the difference between a list and a tuple?**
#    *Answer:* Lists are mutable (can be changed after creation) and use square brackets `[]`. Tuples are immutable and use parentheses `()`. Tuples are generally slightly faster and more memory efficient.
# 2. **What does it mean for a data type to be immutable?**
#    *Answer:* It means its state or contents cannot be altered once it is created. Operations that seem to modify strings or tuples actually create entirely new objects.
# 3. **Can you use a list as a dictionary key? Why or why not?**
#    *Answer:* No, dictionary keys must be hashable and immutable. Since a list is mutable, its hash value would change if its contents changed, breaking the dictionary's internal hash table.
# 4. **How do you convert a list with duplicate elements into a list with only unique elements?**
#    *Answer:* By casting it to a set and back to a list: `unique_list = list(set(original_list))`.
# 5. **What is the difference between `None` and `False` in Python?**
#    *Answer:* `None` is a special constant representing the absence of a value or a null value (type `NoneType`). `False` is a boolean data type.
#
# # Practice Problems
# 1. Create a dictionary storing hyperparameter names as keys and their values as values. Print the keys and values separately.
# 2. Write code that takes a string representing a float (e.g., "3.14"), converts it to a float, multiplies it by 2, and converts it to an integer.
# 3. Given a list of ages `[25, 30, 25, 40, 30, 50]`, use a set to find the unique ages and print them.
# 4. Create a tuple representing the dimensions of an image (width, height, channels). Attempt to change the channels and catch the error.
# 5. Define an empty dictionary and an empty set, verifying their types using `type()`.
#
# # Solutions

# %%
# Solution to Problem 1
hyperparams = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}
print("Keys:", hyperparams.keys())
print("Values:", hyperparams.values())

# %%
# Solution to Problem 2
str_val = "3.14"
f_val = float(str_val)
doubled = f_val * 2
final_int = int(doubled)
print("Resulting integer:", final_int)

# %%
# Solution to Problem 3
ages = [25, 30, 25, 40, 30, 50]
unique_ages = set(ages)
print("Unique ages:", unique_ages)

# %%
# Solution to Problem 4
image_shape = (1920, 1080, 3)
try:
    image_shape[2] = 1
except TypeError as e:
    print(f"Caught error: {e}")

# %%
# Solution to Problem 5
empty_dict = {}
empty_set = set()
print("Type of {}:", type(empty_dict))
print("Type of set():", type(empty_set))

# %% [markdown]
# # Further Reading
# - [Python Built-in Types](https://docs.python.org/3/library/stdtypes.html)
# - [Real Python - Basic Data Types](https://realpython.com/python-data-types/)
# - [Real Python - Dictionaries in Python](https://realpython.com/python-dicts/)
