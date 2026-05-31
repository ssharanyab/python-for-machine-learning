# %% [markdown]
# # Title
# os: Operating System Interfaces
# 
# # Why this matters
# The `os` module allows Python to interact with the underlying operating system. For machine learning, it's vital for setting environment variables (like API keys or CUDA devices), interacting with the shell, and performing legacy path operations when older libraries require them.
# 
# # Learning Objectives
# - Manage environment variables.
# - Execute system commands.
# - Perform directory traversal using `os.walk`.
# - Differentiate between `os` operations and `pathlib` features.
# 
# # Concept Explanation
# The `os` module provides a portable way of using operating system dependent functionality. While `pathlib` is preferred for path manipulations, `os` is indispensable for environment management, process management, and low-level file descriptor operations.
# 
# # Beginner Examples

# %%
import os

# 1. Getting current working directory
cwd = os.getcwd()
print("Current Working Directory:", cwd)

# 2. Listing files in a directory
files = os.listdir('.')
print("Files in current directory:", files[:5])

# 3. Environment Variables
# Setting an environment variable (often used for config/secrets)
os.environ['MY_ML_API_KEY'] = '12345'
print("API Key:", os.environ.get('MY_ML_API_KEY'))

# 4. Creating and removing directories
os.makedirs('temp_dir/nested', exist_ok=True)
os.removedirs('temp_dir/nested')

# %% [markdown]
# # Intermediate Examples

# %%
# 1. Using os.walk for directory traversal
def traverse_directory(path='.'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                print(os.path.join(root, file))
                break # Just print the first one for example
        break

traverse_directory()

# 2. Path manipulations with os.path (Legacy approach)
# Often seen in older ML codebases
base_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
dataset_path = os.path.join(base_path, '..', 'datasets', 'sample.csv')
print("Dataset Path (os.path):", dataset_path)
print("File exists:", os.path.exists(dataset_path))

# 3. Getting file sizes
if os.path.exists(dataset_path):
    size = os.path.getsize(dataset_path)
    print(f"File size: {size} bytes")

# 4. Renaming a file
# with open('old_name.txt', 'w') as f: f.write('data')
# os.rename('old_name.txt', 'new_name.txt')
# os.remove('new_name.txt')

# %% [markdown]
# # Machine Learning Relevance
# You will frequently use `os.environ` to configure environment settings, such as `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` to restrict a deep learning model to a specific GPU. `os.walk` is also useful for recursively discovering thousands of image files in an image classification dataset.
# 
# # Common Mistakes
# - Using `os.path.join` incorrectly by hardcoding backslashes (`\`) instead of letting the module handle the separator.
# - Modifying `os.environ` and expecting it to persist after the Python script exits (it only affects the current process).
# - Failing to handle missing permissions when using `os.remove` or `os.makedirs`.
# 
# # Interview Questions
# 1. What is the difference between `os.listdir()` and `os.walk()`?
# 2. How do you safely retrieve an environment variable without raising an error if it doesn't exist?
# 3. When would you use `os.path.join` over standard string concatenation?
# 4. How can you set a temporary environment variable for a subprocess?
# 5. Which module is generally preferred for path manipulations in modern Python, `os.path` or `pathlib`?
# 
# # Practice Problems
# 1. Print the value of the 'PATH' environment variable.
# 2. Write a script to create a nested directory structure `a/b/c`.
# 3. Use `os.walk` to count the total number of files in a directory.
# 4. Use `os.path.split` to separate a directory path from the file name.
# 5. Check whether a given path is a file or a directory using `os.path`.
# 
# # Solutions

# %%
# Solution 1
print("PATH:", os.environ.get('PATH')[:50], "...")

# Solution 2
os.makedirs('a/b/c', exist_ok=True)
print("Created a/b/c")

# Solution 3
def count_files(start_path):
    total = 0
    for _, _, files in os.walk(start_path):
        total += len(files)
    return total
print(f"Total files in a/b/c: {count_files('a/b/c')}")

# Solution 4
head, tail = os.path.split('/path/to/my_file.txt')
print("Directory:", head)
print("Filename:", tail)

# Solution 5
path_to_check = 'a/b/c'
print("Is directory?", os.path.isdir(path_to_check))
print("Is file?", os.path.isfile(path_to_check))

# Cleanup
os.removedirs('a/b/c')

# %% [markdown]
# # Further Reading
# - [Official Python Documentation for os](https://docs.python.org/3/library/os.html)
