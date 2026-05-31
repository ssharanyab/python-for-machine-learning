# %% [markdown]
# # Python Dataclasses for Machine Learning
# 
# # Why this matters
# In ML experiments, you constantly pass around configuration objects: hyperparameters, file paths, model architectural settings. Writing `__init__`, `__repr__`, and `__eq__` methods for all these configuration objects is tedious. Python 3.7+ introduced `@dataclass`, which automatically generates this boilerplate code, making your data containers clean, readable, and less error-prone.
# 
# # Learning Objectives
# - Understand the purpose and benefits of `dataclasses`.
# - Learn how to define a dataclass.
# - Understand default values and type hinting in dataclasses.
# - Use the `field()` function for advanced attribute configurations.
# 
# # Concept Explanation
# A `dataclass` is a decorator that automatically generates special methods like `__init__()` and `__repr__()` for user-defined classes. They are designed for classes that primarily store data, minimizing boilerplate code. They enforce type hinting, which improves code readability and IDE support.
# 
# # Beginner Examples

# %%
# Example 1: Standard Class vs Dataclass
from dataclasses import dataclass

# The tedious way
class ConfigOld:
    def __init__(self, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size
    def __repr__(self):
        return f"ConfigOld(lr={self.lr}, batch_size={self.batch_size})"

# The Dataclass way
@dataclass
class ConfigNew:
    lr: float
    batch_size: int

c = ConfigNew(0.01, 32)
print(c) # Automatically gets a nice string representation

# %%
# Example 2: Default values
@dataclass
class ModelConfig:
    layers: int = 3
    activation: str = "relu"

print(ModelConfig()) # Uses defaults
print(ModelConfig(layers=5))

# %%
# Example 3: Automatic equality (__eq__)
c1 = ConfigNew(0.1, 16)
c2 = ConfigNew(0.1, 16)
print("Are they equal?", c1 == c2) # True, compares values, not memory address!

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: Frozen dataclasses (Immutable)
@dataclass(frozen=True)
class FixedHyperparams:
    seed: int = 42
    
params = FixedHyperparams()
# params.seed = 99 # This would raise a FrozenInstanceError

# %%
# Example 5: Using field() for mutable defaults
from dataclasses import field
from typing import List

@dataclass
class ExperimentConfig:
    name: str
    # You cannot do `metrics: List[str] = []` directly due to mutable default risks.
    # Instead, use field(default_factory=...)
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1"])

exp = ExperimentConfig("run_1")
print(exp)

# %% [markdown]
# # Machine Learning Relevance
# Libraries like HuggingFace `transformers` extensively use dataclasses (e.g., `TrainingArguments`) to handle the massive number of configuration options required for training Large Language Models. Dataclasses keep configuration structures strict, type-checked, and serializable (easy to convert to JSON for experiment tracking).
# 
# # Common Mistakes
# 1. Forgetting type hints (dataclasses *require* type hints to recognize fields).
# 2. Using mutable defaults (like `[]` or `{}`) directly instead of `default_factory`.
# 3. Putting non-default fields after default fields.
# 4. Confusing `@dataclass` with NamedTuples or Pydantic models (which offer validation).
# 
# # Interview Questions
# 1. What is the primary benefit of using the `@dataclass` decorator?
# 2. What special methods does a dataclass generate automatically?
# 3. Why are type hints mandatory when defining a dataclass?
# 4. How do you handle mutable default arguments (like lists) in a dataclass?
# 5. What does the `frozen=True` argument do in a dataclass?
# 
# # Practice Problems
# 1. Define a `@dataclass` named `Point` with `x` and `y` float coordinates.
# 2. Define a `@dataclass` `TrainingArgs` with `epochs` (int, default 10) and `learning_rate` (float, default 0.001).
# 3. Create two instances of `TrainingArgs` with default values and check if they are equal.
# 4. Define a frozen `@dataclass` called `Constants` containing a `PI` value.
# 5. Define a dataclass `DatasetInfo` with a string `name` and a `features` list using `default_factory`.
# 
# # Solutions

# %%
# Solution 1
@dataclass
class Point:
    x: float
    y: float

# %%
# Solution 2 & 3
@dataclass
class TrainingArgs:
    epochs: int = 10
    learning_rate: float = 0.001

t1, t2 = TrainingArgs(), TrainingArgs()
print(t1 == t2)

# %%
# Solution 4
@dataclass(frozen=True)
class Constants:
    PI: float = 3.14159

# %%
# Solution 5
@dataclass
class DatasetInfo:
    name: str
    features: list = field(default_factory=list)

# %% [markdown]
# # Further Reading
# - Python Dataclasses Official Documentation
# - Comparing Dataclasses vs NamedTuples vs Pydantic
