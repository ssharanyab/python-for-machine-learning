# %% [markdown]
# # Title
# Pathlib: Object-Oriented Filesystem Paths
# 
# # Why this matters
# In machine learning, you constantly deal with file paths to load datasets, save models, and organize experiments. Pathlib provides an intuitive, object-oriented way to handle paths across different operating systems (Windows, Linux, macOS), ensuring your code is cross-platform compatible.
# 
# # Learning Objectives
# - Understand the advantages of Pathlib over older modules.
# - Learn how to construct, manipulate, and query paths.
# - Discover how to iterate over directories and filter files.
# - Apply Pathlib to load and save datasets robustly.
# 
# # Concept Explanation
# `pathlib` introduces the `Path` class. Instead of manipulating strings representing paths, you create `Path` objects. This allows you to use methods and properties to interact with paths cleanly, such as `path.parent`, `path.name`, or the `/` operator for joining paths.
# 
# # Beginner Examples

# %%
from pathlib import Path

# 1. Creating a Path object
current_dir = Path('.')
print("Current directory:", current_dir.absolute())

# 2. Joining paths
data_dir = current_dir / '..' / 'datasets'
print("Data directory:", data_dir.resolve())

# 3. Checking if a path exists
print("Exists?", data_dir.exists())

# 4. Getting file metadata
file_path = data_dir / 'sample.csv'
print("File name:", file_path.name)
print("File suffix:", file_path.suffix)

# %% [markdown]
# # Intermediate Examples

# %%
# 1. Iterating over directories
if data_dir.exists():
    for file in data_dir.iterdir():
        print("Found:", file.name)

# 2. Globbing (Finding specific files)
if data_dir.exists():
    csv_files = list(data_dir.glob('*.csv'))
    print("CSV Files:", csv_files)

# 3. Reading/Writing text directly
temp_file = Path('temp.txt')
temp_file.write_text("Machine Learning Data")
print("Read content:", temp_file.read_text())
temp_file.unlink() # Delete file

# 4. Creating directories
new_dir = Path('my_experiments')
new_dir.mkdir(exist_ok=True)
new_dir.rmdir() # Remove directory

# %% [markdown]
# # Machine Learning Relevance
# When setting up an ML pipeline, you often need to define paths for raw data, processed data, and model checkpoints. Pathlib makes it easy to construct these paths relative to your script's location, ensuring your project runs seamlessly regardless of where it's executed or on which OS.
# 
# # Common Mistakes
# - Using string concatenation instead of the `/` operator, which can cause OS-specific path issues.
# - Forgetting to call `.resolve()` when an absolute path is required by legacy libraries.
# - Not handling `FileNotFoundError` when attempting to read from a Path that does not exist.
# 
# # Interview Questions
# 1. How does `pathlib` differ from the `os.path` module?
# 2. How do you concatenate two paths using `pathlib`?
# 3. Explain how to find all `.png` files in a directory and its subdirectories using `pathlib`.
# 4. How can you get the parent directory of a given file path?
# 5. Is a `Path` object mutable?
# 
# # Practice Problems
# 1. Create a `Path` object for the current script and print its absolute path.
# 2. Construct a path to `../datasets/sample.csv` and check if it exists.
# 3. Create a new directory named `outputs`, write "test" to a file inside it, then delete both the file and the directory.
# 4. Write a function that takes a directory path and returns a list of all `.csv` files.
# 5. Extract the file extension from a given file path.
# 
# # Solutions

# %%
# Solution 1
p = Path(__file__) if '__file__' in globals() else Path('.').resolve()
print("Absolute script path:", p)

# Solution 2
dataset_path = Path('..') / 'datasets' / 'sample.csv'
print(f"Exists: {dataset_path.exists()}")

# Solution 3
out_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
out_file = out_dir / 'test.txt'
out_file.write_text("test")
print("Content:", out_file.read_text())
out_file.unlink()
out_dir.rmdir()

# Solution 4
def get_csvs(dir_path):
    p = Path(dir_path)
    return list(p.glob('*.csv')) if p.exists() else []
print("CSV files:", get_csvs(dataset_path.parent))

# Solution 5
sample_path = Path('/path/to/my_model.pkl')
print("Extension:", sample_path.suffix)

# %% [markdown]
# # Further Reading
# - [Official Python Documentation for pathlib](https://docs.python.org/3/library/pathlib.html)
# - Real Python guide on Pathlib
