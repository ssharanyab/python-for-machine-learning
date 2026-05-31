# %% [markdown]
# # Python Objects for Machine Learning
# 
# # Why this matters
# While a Class is a blueprint, an **Object** is the actual working mechanism in memory. In Machine Learning, when you train a model, you are instantiating an object and modifying its internal state (fitting weights). When you load a dataset, it's an object. Understanding how to instantiate and interact with objects is crucial for applying ML pipelines.
# 
# # Learning Objectives
# - Understand how to instantiate an object from a class.
# - Learn to access object attributes and call object methods.
# - Understand object mutability and identity in Python.
# - See how objects represent stateful entities in ML.
# 
# # Concept Explanation
# An **Object** is an instance of a Class. It occupies memory and has a specific state defined by its attributes. Multiple objects can be instantiated from the same class, each maintaining its own independent state.
# 
# # Beginner Examples

# %%
# First, let's define a class to use
class ModelConfig:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

# Example 1: Instantiating an object
config1 = ModelConfig(0.01, 100)

# %%
# Example 2: Accessing object attributes
print(f"Learning Rate: {config1.learning_rate}")

# %%
# Example 3: Modifying object state
config1.epochs = 200
print(f"Updated Epochs: {config1.epochs}")

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: Multiple independent objects
config2 = ModelConfig(0.001, 50)
print(f"Config1 LR: {config1.learning_rate}, Config2 LR: {config2.learning_rate}")

# %%
# Example 5: Objects interacting
class ModelTrainer:
    def __init__(self, config):
        self.config = config # Storing an object inside another object
        
    def get_training_info(self):
        return f"Training for {self.config.epochs} epochs at LR {self.config.learning_rate}"

trainer = ModelTrainer(config1)
print(trainer.get_training_info())

# %% [markdown]
# # Machine Learning Relevance
# In `scikit-learn`, `clf = RandomForestClassifier()` creates an object. Before calling `clf.fit(X, y)`, the object is untrained. After calling `.fit()`, the object's internal state is updated with the learned decision trees. Managing these objects, saving them (e.g., via `joblib` or `pickle`), and loading them for inference is the standard ML deployment lifecycle.
# 
# # Common Mistakes
# 1. Confusing the class name with the object variable name.
# 2. Forgetting the parentheses when instantiating an object `obj = MyClass` vs `obj = MyClass()`.
# 3. Expecting changes in one object to affect another object of the same class.
# 4. Not realizing that passing objects to functions passes them by reference.
# 
# # Interview Questions
# 1. How do you instantiate an object from a class in Python?
# 2. Explain object identity and how the `id()` function works.
# 3. What is the difference between an object and a class?
# 4. How can you find the class of an object at runtime?
# 5. Are objects in Python mutable by default?
# 
# # Practice Problems
# 1. Create a `DataSample` class with `features` and `label`. Instantiate two different objects.
# 2. Write a function that takes a `ModelConfig` object and prints its attributes.
# 3. Demonstrate that two objects of the same class have different memory addresses (using `id()`).
# 4. Create an object and dynamically add a new attribute to it that wasn't defined in the class.
# 5. Create a list containing 3 different `ModelConfig` objects.
# 
# # Solutions

# %%
# Solution 1
class DataSample:
    def __init__(self, features, label):
        self.features = features
        self.label = label

sample1 = DataSample([1.0, 2.0], 0)
sample2 = DataSample([3.0, 4.0], 1)

# %%
# Solution 2
def print_config(config):
    print(f"LR: {config.learning_rate}, Epochs: {config.epochs}")

print_config(config1)

# %%
# Solution 3
print(id(config1) == id(config2)) # False

# %%
# Solution 4
config1.optimizer = "adam"
print(config1.optimizer)

# %%
# Solution 5
configs = [ModelConfig(0.1, 10), ModelConfig(0.01, 20), ModelConfig(0.001, 30)]

# %% [markdown]
# # Further Reading
# - Python Object Model documentation
# - "Fluent Python" - Data model chapter
# - Scikit-learn estimator API (understanding how estimators act as objects)
