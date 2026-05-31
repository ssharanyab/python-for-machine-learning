# %% [markdown]
# # Environment Setup for Machine Learning
#
# # Why this matters
# Setting up a robust environment is the foundation of any Machine Learning project. Without reproducible environments, models that work on your machine may fail in production or on a colleague's computer. Proper dependency management prevents "dependency hell" and ensures your workflows are stable.
#
# # Learning Objectives
# 1. Understand how to install Python and manage environments.
# 2. Learn how to use pip and conda for package management.
# 3. Know how to verify the installation of essential ML libraries.
# 4. Grasp the importance of virtual environments in ML projects.
#
# # Concept Explanation
# A Python environment consists of a specific version of Python and a set of installed packages. Virtual environments isolate these setups per project so that dependencies do not conflict. For ML, we often use tools like `venv`, `pip`, or `conda` (part of Anaconda/Miniconda) because they handle complex C/C++ dependencies (like those in NumPy or TensorFlow) effectively.
#
# # Beginner Examples
# Let's verify our Python installation and explore the built-in sys module.

# %%
import sys
print("Python version:")
print(sys.version)
print("\nVersion info:")
print(sys.version_info)

# %% [markdown]
# # Intermediate Examples
# Here we will simulate checking for essential Machine Learning packages. We'll use the `importlib` library to verify if packages like numpy, pandas, and sklearn are installed.

# %%
import importlib

essential_packages = ['numpy', 'pandas', 'sklearn', 'matplotlib']
missing_packages = []

for package in essential_packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package} is installed.")
    except ImportError:
        print(f"❌ {package} is missing.")
        missing_packages.append(package)

if missing_packages:
    print(f"Please install missing packages using: pip install {' '.join(missing_packages)}")

# %% [markdown]
# # Machine Learning Relevance
# In Machine Learning, environments dictate reproducibility. When you publish a model, you also publish a `requirements.txt` or `environment.yml` file. If the environment isn't strictly controlled, your scikit-learn version might differ from the deployment server, causing your model loading (`pickle` or `joblib`) to fail catastrophically.
#
# # Common Mistakes
# 1. Installing packages globally instead of within a virtual environment.
# 2. Mixing `pip` and `conda` indiscriminately, leading to broken environments.
# 3. Forgetting to update or freeze the `requirements.txt` file before sharing code.
# 4. Ignoring Python version requirements for specific ML libraries (e.g., PyTorch might not support the absolute newest Python release immediately).
#
# # Interview Questions
# 1. **What is a virtual environment and why is it necessary in Python?**
#    *Answer:* A virtual environment is an isolated directory containing a specific Python version and packages. It's necessary to prevent dependency conflicts between different projects.
# 2. **Explain the difference between `pip` and `conda`.**
#    *Answer:* `pip` is the Python Package Installer, fetching packages from PyPI. `conda` is a cross-language package manager that also manages Python itself and resolves complex non-Python dependencies (like C++ libraries often used in ML).
# 3. **How do you generate a `requirements.txt` file?**
#    *Answer:* By running `pip freeze > requirements.txt` in the terminal.
# 4. **What is the difference between a `requirements.txt` and a `setup.py`?**
#    *Answer:* `requirements.txt` is used to replicate a specific environment, usually for a deployed app or service. `setup.py` is used when building a library to distribute, defining abstract dependencies.
# 5. **If a model fails to load in production with a "module not found" error despite being trained successfully locally, what is the most likely cause?**
#    *Answer:* An environment mismatch where the required dependency was not installed or recorded in the production environment setup files.
#
# # Practice Problems
# 1. Write a script that checks if the current Python version is at least 3.8, and raises an Exception if it is not.
# 2. Create a list of hypothetical ML package names. Write a loop that tries to import them, capturing the specific exception if they don't exist, and gracefully logs the error.
# 3. Write a function that reads a hypothetical `requirements.txt` format string and extracts only the package names (ignoring versions like `==1.0.0`).
# 4. Simulate a dependency version checker: given a dictionary of installed packages and their versions, verify if `numpy` is >= `1.19.0`.
# 5. Write code that prints out the current working directory and the path to the Python executable being used.
#
# # Solutions

# %%
# Solution to Problem 1
import sys
if sys.version_info < (3, 8):
    raise Exception("Python 3.8 or higher is required.")
print("Python version is valid.")

# %%
# Solution to Problem 2
packages_to_test = ['os', 'sys', 'fake_ml_lib', 'tensor_fake']
for pkg in packages_to_test:
    try:
        __import__(pkg)
        print(f"Successfully imported {pkg}")
    except ImportError as e:
        print(f"Log: Failed to import {pkg}. Reason: {e}")

# %%
# Solution to Problem 3
requirements = "numpy==1.21.0\npandas>=1.3.0\nscikit-learn\nmatplotlib"
def extract_names(req_string):
    packages = []
    for line in req_string.split('\n'):
        name = line.split('==')[0].split('>=')[0].split('<=')[0]
        packages.append(name)
    return packages
print("Extracted packages:", extract_names(requirements))

# %%
# Solution to Problem 4
installed = {'numpy': '1.18.5', 'pandas': '1.3.1'}
def check_numpy_version(installed_dict):
    if 'numpy' in installed_dict:
        version = installed_dict['numpy'].split('.')
        if int(version[0]) >= 1 and int(version[1]) >= 19:
            return True
    return False
print("Is NumPy >= 1.19.0?", check_numpy_version(installed))

# %%
# Solution to Problem 5
import os
print("Current Working Directory:", os.getcwd())
print("Python Executable:", sys.executable)

# %% [markdown]
# # Further Reading
# - [Python Virtual Environments - Official Documentation](https://docs.python.org/3/tutorial/venv.html)
# - [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
# - [Pip Documentation](https://pip.pypa.io/en/stable/)
