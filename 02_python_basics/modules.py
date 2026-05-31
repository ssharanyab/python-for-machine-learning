# %% [markdown]
# # Modules and Packages
#
# ## Why this matters
# Machine learning relies entirely on external libraries (modules/packages) like NumPy, Pandas, Scikit-Learn, and PyTorch. Understanding how to import, alias, and organize code into modules is fundamental to writing clean, maintainable ML code.
#
# ## Learning Objectives
# - Understand the difference between a module and a package.
# - Learn how to import entire modules, specific functions, and use aliases.
# - Understand the purpose of `__name__ == "__main__"`.
# - Learn how Python searches for modules (sys.path).
#
# ## Concept Explanation
# - **Module:** A single Python file containing variables, functions, and classes.
# - **Package:** A directory of Python modules containing an `__init__.py` file (though optional in Python 3.3+).
# Use `import` to bring these resources into your current script.
#
# ## Beginner Examples

# %%
# Example 1: Importing a built-in module
import math

print(f"Pi is roughly: {math.pi}")
print(f"Square root of 16 is: {math.sqrt(16)}")

# %%
# Example 2: Importing specific functions
from datetime import datetime, timedelta

now = datetime.now()
print(f"Current Time: {now}")

# %%
# Example 3: Importing with aliases
import random as rnd

print(f"Random number between 1 and 10: {rnd.randint(1, 10)}")

# %% [markdown]
# ## Intermediate Examples

# %%
# Example 4: Creating and importing your own module
# (Pretend we have a file named my_ml_tools.py with a function calculate_mse)
# import my_ml_tools
# mse = my_ml_tools.calculate_mse(y_true, y_pred)

# For demonstration, we'll write a temporary module
with open("temp_math.py", "w") as f:
    f.write("def add(a, b): return a + b\n")

import temp_math
print(f"Using custom module: {temp_math.add(5, 7)}")

# Clean up
import os
os.remove("temp_math.py")

# %%
# Example 5: Exploring module contents using dir()
import json
print("Functions in json module:")
print([x for x in dir(json) if not x.startswith('_')][:5])

# %%
# Example 6: The __name__ == "__main__" block
def main():
    print("This runs only if the script is executed directly.")

if __name__ == "__main__":
    main()

# %% [markdown]
# ## Machine Learning Relevance
# In data science, you will almost universally use these standard aliases:
# `import numpy as np`
# `import pandas as pd`
# `import matplotlib.pyplot as plt`

# %%
# ML Example: Simulating standard imports
import math as np_sim  # Pretend math is numpy for this environment
values = [1, 2, 3, 4]
# Pretend np_sim.exp is a vectorized numpy function
exp_values = [np_sim.exp(x) for x in values]
print(f"Exponential values: {exp_values}")

# %% [markdown]
# ## Common Mistakes
# 1. **Circular Imports:** Module A imports Module B, and Module B imports Module A.
# 2. **Shadowing Standard Modules:** Naming your script `math.py`. When you try to `import math`, Python imports your script instead of the built-in module.
# 3. **Import `*`:** Using `from module import *`. This pollutes the global namespace and makes it impossible to know where functions came from.

# %% [markdown]
# ## Interview Questions
# 1. What is the difference between a module, a package, and a library?
# 2. Why is `from module import *` considered bad practice?
# 3. Explain what `if __name__ == "__main__":` does.
# 4. How does Python find the modules to import (what is `sys.path`)?
# 5. What is the standard alias used when importing pandas and numpy?

# %% [markdown]
# ## Practice Problems
# 1. Import the `os` module and print the current working directory.
# 2. Import the `Counter` class from the `collections` module and use it to count the occurrences of letters in "machinelearning".
# 3. Import `statistics` using an alias `stats`, and calculate the median of `[1, 5, 2, 4, 3]`.
# 4. Write a string of Python code that defines a simple function. Save it to `temp_module.py`. Import it, call the function, and then delete the file using `os`.
# 5. Import the `sys` module and print the first two paths in `sys.path`.

# %% [markdown]
# ## Solutions

# %%
# Solution 1: OS CWD
import os
print(os.getcwd())

# %%
# Solution 2: Collections Counter
from collections import Counter
counts = Counter("machinelearning")
print(counts)

# %%
# Solution 3: Statistics alias
import statistics as stats
median_val = stats.median([1, 5, 2, 4, 3])
print(median_val)

# %%
# Solution 4: Custom module workflow
with open("my_test_module.py", "w") as f:
    f.write("def greet(): return 'Hello from module!'\n")

import my_test_module
print(my_test_module.greet())
os.remove("my_test_module.py")

# %%
# Solution 5: Sys path
import sys
print(sys.path[:2])

# %% [markdown]
# ## Further Reading
# - Python Modules Documentation
# - Real Python: Python Modules and Packages
