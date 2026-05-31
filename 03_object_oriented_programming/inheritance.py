# %% [markdown]
# # Python Inheritance for Machine Learning
# 
# # Why this matters
# In Machine Learning ecosystems, we often have different types of models (e.g., Decision Trees, SVMs, Neural Networks) that share common functionalities like `fit`, `predict`, and `evaluate`. **Inheritance** allows us to create a base class with shared logic and derive specialized classes from it, reducing code duplication and enforcing a consistent API.
# 
# # Learning Objectives
# - Understand the concept of Inheritance in OOP.
# - Learn how to define a subclass that inherits from a superclass (base class).
# - Understand method overriding.
# - Learn how to use the `super()` function.
# 
# # Concept Explanation
# **Inheritance** is a mechanism where a new class (subclass or child class) derives attributes and methods from an existing class (superclass or parent class). The subclass can add new attributes/methods or override existing ones to specialize behavior.
# 
# # Beginner Examples

# %%
# Example 1: Base Class
class MLModel:
    def __init__(self, name):
        self.name = name
        self.is_trained = False
        
    def fit(self, X, y):
        print(f"Training {self.name}...")
        self.is_trained = True

# %%
# Example 2: Subclass inheriting from MLModel
class LinearRegression(MLModel):
    pass # Inherits everything, adds nothing new

model = LinearRegression("LinearRegression_v1")
model.fit([1,2], [3,4]) # Calling inherited method

# %%
# Example 3: Overriding a method
class DummyClassifier(MLModel):
    def fit(self, X, y):
        print(f"DummyClassifier ignores training data.")
        self.is_trained = True

dummy = DummyClassifier("Dummy")
dummy.fit([], [])

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: Using super() to extend __init__
class RandomForest(MLModel):
    def __init__(self, name, n_estimators):
        super().__init__(name) # Call parent's __init__
        self.n_estimators = n_estimators

rf = RandomForest("RF_Classifier", 100)
print(f"Model: {rf.name}, Trees: {rf.n_estimators}, Trained: {rf.is_trained}")

# %%
# Example 5: Multi-level inheritance
class GradientBoosting(RandomForest):
    def __init__(self, name, n_estimators, learning_rate):
        super().__init__(name, n_estimators)
        self.learning_rate = learning_rate

gb = GradientBoosting("GB_Classifier", 50, 0.1)
print(f"Model: {gb.name}, LR: {gb.learning_rate}")

# %% [markdown]
# # Machine Learning Relevance
# Scikit-learn uses inheritance extensively. Classes like `BaseEstimator`, `ClassifierMixin`, and `RegressorMixin` provide baseline functionality. A custom model usually inherits from `BaseEstimator` to automatically get capabilities like `get_params()` and `set_params()` required for hyperparameter tuning (GridSearchCV). Deep learning frameworks like PyTorch require custom network layers to inherit from `torch.nn.Module`.
# 
# # Common Mistakes
# 1. Forgetting to call `super().__init__()` when defining an `__init__` in the subclass.
# 2. Overriding methods but changing the method signature (arguments) unexpectedly, breaking the API.
# 3. Creating inheritance hierarchies that are too deep and difficult to maintain.
# 4. Misusing inheritance when composition (having a class as an attribute) would be better.
# 
# # Interview Questions
# 1. What is inheritance and why is it useful in software design?
# 2. How do you implement inheritance in Python?
# 3. What does the `super()` function do?
# 4. Can a subclass override a method of the superclass? How?
# 5. What is multiple inheritance, and does Python support it?
# 
# # Practice Problems
# 1. Create a base `Dataset` class and a subclass `ImageDataset`.
# 2. In `ImageDataset`, override the `__init__` to add an `image_size` attribute using `super()`.
# 3. Create a base `LossFunction` class with a `compute` method that raises `NotImplementedError`.
# 4. Create a subclass `MSELoss` inheriting from `LossFunction` that overrides `compute`.
# 5. Instantiate `MSELoss` and call `compute`.
# 
# # Solutions

# %%
# Solution 1 & 2
class Dataset:
    def __init__(self, path):
        self.path = path

class ImageDataset(Dataset):
    def __init__(self, path, image_size):
        super().__init__(path)
        self.image_size = image_size

# %%
# Solution 3
class LossFunction:
    def compute(self, y_true, y_pred):
        raise NotImplementedError("Subclasses must implement compute()")

# %%
# Solution 4 & 5
class MSELoss(LossFunction):
    def compute(self, y_true, y_pred):
        # Simplified mock calculation
        return 0.5

mse = MSELoss()
print("MSE Loss:", mse.compute([], []))

# %% [markdown]
# # Further Reading
# - Python documentation on Inheritance
# - The Liskov Substitution Principle (SOLID principles)
# - Scikit-learn Developer Guide on writing custom estimators
