# %% [markdown]
# # Python Encapsulation for Machine Learning
# 
# # Why this matters
# **Encapsulation** is the bundling of data and methods that operate on that data within a single unit (the class), and restricting direct access to some of the object's components. In ML, this is vital for data integrity. You don't want external code accidentally modifying a model's learned weights or bypassing validation checks when setting hyperparameters.
# 
# # Learning Objectives
# - Understand the concept of Encapsulation and data hiding.
# - Learn conventions for protected (`_attr`) and private (`__attr`) attributes in Python.
# - Understand how to use getter and setter methods.
# - Use the `@property` decorator for pythonic encapsulation.
# 
# # Concept Explanation
# Encapsulation protects an object's internal state. Python doesn't have strict access modifiers like `public` or `private` (as in Java/C++). Instead, it relies on naming conventions and name mangling. A single underscore `_` implies an attribute is for internal use (protected). A double underscore `__` invokes name mangling to make it harder to access from outside (private).
# 
# # Beginner Examples

# %%
# Example 1: Public vs Protected attributes
class DataPipeline:
    def __init__(self):
        self.name = "Standard Pipeline" # Public
        self._cache = {}                # Protected (convention)

pipeline = DataPipeline()
print(pipeline.name)
print(pipeline._cache) # Accessible, but discouraged

# %%
# Example 2: "Private" attributes (Name Mangling)
class SecureModel:
    def __init__(self):
        self.__weights = [0.5, 0.5] # Private

model = SecureModel()
# print(model.__weights) # This will raise an AttributeError!

# %%
# Example 3: Accessing private attributes via methods
class SecureModelAccess(SecureModel):
    def __init__(self):
        self.__weights = [0.5, 0.5]
        
    def get_weights(self):
        return self.__weights

model2 = SecureModelAccess()
print(model2.get_weights())

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: Using @property for pythonic encapsulation
class Hyperparameters:
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        
    @property
    def learning_rate(self):
        """Getter for learning rate."""
        return self._learning_rate
        
    @learning_rate.setter
    def learning_rate(self, value):
        """Setter with validation."""
        if value <= 0 or value >= 1:
            raise ValueError("Learning rate must be between 0 and 1.")
        self._learning_rate = value

params = Hyperparameters(0.01)
print(params.learning_rate) # Calls the getter
params.learning_rate = 0.1  # Calls the setter

# %%
# Example 5: Read-only properties
class ModelMetrics:
    def __init__(self):
        self._accuracy = 0.95
        
    @property
    def accuracy(self):
        # No setter defined, so this is read-only from the outside
        return self._accuracy

metrics = ModelMetrics()
print(metrics.accuracy)
# metrics.accuracy = 0.99 # Raises AttributeError

# %% [markdown]
# # Machine Learning Relevance
# Deep learning libraries like PyTorch extensively use properties to guard sensitive states. Scikit-learn estimators use a trailing underscore convention (`clf.coef_`) to denote attributes that are estimated from the data during `.fit()`, conceptually separating user-provided hyperparameters from model-learned parameters.
# 
# # Common Mistakes
# 1. Overusing getters/setters in Python when simple public attributes suffice (not Pythonic).
# 2. Believing `__private` makes an attribute truly inaccessible (it's just name-mangled to `_ClassName__private`).
# 3. Forgetting to use the `@property` decorator and writing Java-style `get_value()` methods everywhere.
# 4. Modifying `_protected` attributes from outside the class, violating the API contract.
# 
# # Interview Questions
# 1. What does encapsulation mean in Object-Oriented Programming?
# 2. How does Python indicate that an attribute is protected or private?
# 3. What is name mangling in Python, and how does it relate to encapsulation?
# 4. What is the purpose of the `@property` decorator?
# 5. Why might you use a setter method instead of allowing direct attribute access?
# 
# # Practice Problems
# 1. Create a `Dataset` class with a `_filepath` protected attribute.
# 2. Create a `NeuralNetwork` class with a `__layers` private list attribute.
# 3. Create a `TrainingConfig` class with a `batch_size` property.
# 4. Add a setter for `batch_size` that raises a ValueError if the size is not a positive integer.
# 5. Access the name-mangled `__layers` attribute of your `NeuralNetwork` instance from outside the class (for educational purposes).
# 
# # Solutions

# %%
# Solution 1
class Dataset:
    def __init__(self):
        self._filepath = "data.csv"

# %%
# Solution 2
class NeuralNetwork:
    def __init__(self):
        self.__layers = [10, 20, 1]

# %%
# Solution 3 & 4
class TrainingConfig:
    def __init__(self, batch_size):
        self.batch_size = batch_size # uses setter
        
    @property
    def batch_size(self):
        return self._batch_size
        
    @batch_size.setter
    def batch_size(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Batch size must be positive int")
        self._batch_size = value

# %%
# Solution 5
nn = NeuralNetwork()
print(nn._NeuralNetwork__layers) # Accessing name-mangled attribute

# %% [markdown]
# # Further Reading
# - Python docs on Private Variables
# - "Effective Python" guidelines on using properties
