# %% [markdown]
# # Python Classes for Machine Learning
# 
# # Why this matters
# In Machine Learning, real-world problems are complex. We need a way to organize code, group related data (attributes) and behaviors (methods) together to build robust systems. Classes provide the blueprint for creating these organized structures, whether it's a dataset handler, a model trainer, or a feature engineering pipeline.
# 
# # Learning Objectives
# - Understand the concept of a Class in Python.
# - Learn how to define a Class and the `__init__` constructor.
# - Differentiate between class attributes and instance attributes.
# - Understand how to add methods to a class.
# 
# # Concept Explanation
# A **Class** is a blueprint for creating objects. It defines a set of attributes that will characterize any object that is instantiated from this class. 
# 
# Think of a Class as a cookie cutter, and the objects (instances) as the cookies. All cookies made with the same cutter have the same basic shape, but they might have different decorations (attributes).
# 
# # Beginner Examples

# %%
# Example 1: A simple empty class
class SimpleModel:
    pass

# %%
# Example 2: Class with a constructor
class Dataset:
    def __init__(self, name):
        self.name = name

# %%
# Example 3: Adding methods
class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def accuracy(self):
        correct = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p)
        return correct / len(self.y_true)

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: Class variables vs Instance variables
class MLFramework:
    # Class variable
    version = "1.0"
    
    def __init__(self, name):
        # Instance variable
        self.name = name

# %%
# Example 5: A basic Neural Network Layer class blueprint
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [[0.1] * output_size for _ in range(input_size)] # Simplified weights
        
    def forward_pass(self, inputs):
        # Simplified forward pass (dot product placeholder)
        return [sum(i * w[0] for i, w in zip(inputs, self.weights))]

# %% [markdown]
# # Machine Learning Relevance
# Classes are the backbone of libraries like Scikit-Learn (where every model like `RandomForestClassifier` is a class) and PyTorch (where every neural network module is a class inheriting from `nn.Module`). Organizing code into classes allows for modularity, reusability, and encapsulation of state (like learned weights).
# 
# # Common Mistakes
# 1. Forgetting `self` in method definitions.
# 2. Modifying class variables when intending to modify instance variables.
# 3. Putting too much logic in `__init__` instead of dedicated methods.
# 4. Confusing class definition with object instantiation.
# 
# # Interview Questions
# 1. What is the difference between a class and an object?
# 2. What is the purpose of the `__init__` method in Python?
# 3. What is the `self` parameter, and why is it necessary?
# 4. How do class attributes differ from instance attributes?
# 5. Can you define a class without an `__init__` method? What happens if you do?
# 
# # Practice Problems
# 1. Create a `DataLoader` class that takes a list of data points and a batch size, and has a method to return the total number of batches.
# 2. Create a `LinearRegression` class with an `__init__` method that initializes `weights` and `bias` to None.
# 3. Create a `FeatureScaler` class that takes a `min_val` and `max_val` as instance attributes.
# 4. Create a `Logger` class with a class attribute `log_level` and an instance attribute `name`.
# 5. Create a `TextProcessor` class that takes a string in `__init__` and has a method to return the lowercase version.
# 
# # Solutions

# %%
# Solution 1
class DataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        
    def num_batches(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

# %%
# Solution 2
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

# %%
# Solution 3
class FeatureScaler:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

# %%
# Solution 4
class Logger:
    log_level = "INFO"
    def __init__(self, name):
        self.name = name

# %%
# Solution 5
class TextProcessor:
    def __init__(self, text):
        self.text = text
        
    def to_lower(self):
        return self.text.lower()

# %% [markdown]
# # Further Reading
# - Python Official Documentation on Classes
# - "Fluent Python" by Luciano Ramalho
# - Scikit-Learn API design principles
