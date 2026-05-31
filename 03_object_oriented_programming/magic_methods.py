# %% [markdown]
# # Python Magic Methods (Dunder Methods) for Machine Learning
# 
# # Why this matters
# Magic methods (or "dunder" - double underscore methods) allow your custom objects to integrate seamlessly with standard Python operators and built-in functions. In ML, you might want to add two vectors (using `+`), get the length of a dataset (using `len()`), or index a batch of data (using `dataset[i]`). Magic methods make this intuitive syntax possible.
# 
# # Learning Objectives
# - Understand what magic methods are and why they are called "dunder" methods.
# - Implement `__str__` and `__repr__` for clear object representation.
# - Use `__len__` and `__getitem__` to create custom iterable datasets.
# - Overload math operators like `__add__` and `__mul__`.
# 
# # Concept Explanation
# Magic methods are special methods with double underscores at the beginning and end (e.g., `__init__`, `__len__`). Python calls these methods automatically under the hood when certain operations are performed on objects of the class. 
# 
# # Beginner Examples

# %%
# Example 1: __str__ and __repr__
class ModelStats:
    def __init__(self, acc, loss):
        self.acc = acc
        self.loss = loss
        
    def __str__(self):
        # Human-readable string (used by print)
        return f"Model Status: Acc={self.acc:.2f}, Loss={self.loss:.2f}"
        
    def __repr__(self):
        # Unambiguous string (used by debugger/REPL)
        return f"ModelStats(acc={self.acc}, loss={self.loss})"

stats = ModelStats(0.95, 0.1)
print(stats)
print(repr(stats))

# %%
# Example 2: __len__
class TextCorpus:
    def __init__(self, documents):
        self.documents = documents
        
    def __len__(self):
        # Allows usage of len() function
        return len(self.documents)

corpus = TextCorpus(["doc1", "doc2", "doc3"])
print("Corpus size:", len(corpus))

# %%
# Example 3: __getitem__ for indexing
class CustomDataset:
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        # Allows array-style indexing
        return self.data[index]

dataset = CustomDataset([10, 20, 30])
print("First item:", dataset[0])

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: Operator overloading (__add__)
class Vector:
    def __init__(self, elements):
        self.elements = elements
        
    def __add__(self, other):
        if len(self.elements) != len(other.elements):
            raise ValueError("Vectors must be same length")
        return Vector([a + b for a, b in zip(self.elements, other.elements)])
        
    def __repr__(self):
        return f"Vector({self.elements})"

v1 = Vector([1, 2])
v2 = Vector([3, 4])
print("Vector Addition:", v1 + v2)

# %%
# Example 5: Callable objects (__call__)
class ActivationFunction:
    def __call__(self, x):
        # Makes the object callable like a function
        return max(0, x) # ReLU

relu = ActivationFunction()
print("ReLU(-5):", relu(-5))
print("ReLU(5):", relu(5))

# %% [markdown]
# # Machine Learning Relevance
# PyTorch's `Dataset` class relies entirely on overriding `__len__` and `__getitem__` to create data loaders that can iterate over massive datasets in batches. Neural network layers in PyTorch implement `__call__` (via a `forward` pass), allowing you to pass data through a layer using syntax like `output = layer(input)`.
# 
# # Common Mistakes
# 1. Implementing `__str__` but forgetting `__repr__` (which is often more useful for debugging).
# 2. Returning the wrong type from magic methods (e.g., `__len__` must return an integer).
# 3. Not handling exceptions in operator overloading (e.g., adding objects of incompatible types).
# 4. Confusing `__getattr__` and `__getattribute__`.
# 
# # Interview Questions
# 1. What are magic methods or dunder methods in Python?
# 2. What is the difference between `__str__` and `__repr__`?
# 3. How would you make an object iterable in Python using magic methods?
# 4. What does the `__call__` method do?
# 5. How does Python implement operator overloading?
# 
# # Practice Problems
# 1. Create a `ConfusionMatrix` class and implement `__str__` to print it beautifully.
# 2. Create a `TimeSeries` class that implements `__len__`.
# 3. Add `__getitem__` to the `TimeSeries` class.
# 4. Create a `WeightMatrix` class and implement `__mul__` for scalar multiplication.
# 5. Implement `__eq__` on a `HyperparamConfig` class so two configs with the same settings evaluate to `True` using `==`.
# 
# # Solutions

# %%
# Solution 1
class ConfusionMatrix:
    def __init__(self, tp, fp, tn, fn):
        self.tp, self.fp, self.tn, self.fn = tp, fp, tn, fn
    def __str__(self):
        return f"TP:{self.tp} FP:{self.fp}\nFN:{self.fn} TN:{self.tn}"

# %%
# Solution 2 & 3
class TimeSeries:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# %%
# Solution 4
class WeightMatrix:
    def __init__(self, w):
        self.w = w
    def __mul__(self, scalar):
        return WeightMatrix([x * scalar for x in self.w])

# %%
# Solution 5
class HyperparamConfig:
    def __init__(self, lr, batch):
        self.lr = lr
        self.batch = batch
    def __eq__(self, other):
        return self.lr == other.lr and self.batch == other.batch

# %% [markdown]
# # Further Reading
# - Python Data Model (Documentation on Special Methods)
# - Implementing Custom PyTorch Datasets
