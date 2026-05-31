# %% [markdown]
# # Title
# CSV: Handling Tabular Data
# 
# # Why this matters
# Comma-Separated Values (CSV) files are the lingua franca of machine learning datasets. While libraries like Pandas are often used to process CSVs, understanding the built-in `csv` module is crucial for lightweight data ingestion, memory-efficient streaming, and debugging formatting issues.
# 
# # Learning Objectives
# - Read and write CSV files using the built-in `csv` module.
# - Understand dialects, delimiters, and quote characters.
# - Work with `csv.DictReader` and `csv.DictWriter` for structured parsing.
# - Efficiently process large CSV files line-by-line.
# 
# # Concept Explanation
# The `csv` module provides classes to read and write tabular data. It handles the nuances of escaping delimiters and quoting strings, which is notoriously difficult to do correctly with simple string splitting. `DictReader` maps the information in each row to a dictionary whose keys are given by the first row.
# 
# # Beginner Examples

# %%
import csv
from pathlib import Path

# Create a sample dataset for demonstration
dataset_path = Path('sample_data.csv')
sample_data = [
    ['feature_1', 'feature_2', 'label'],
    [1.5, 2.3, 0],
    [3.1, 4.2, 1],
    [0.9, 1.1, 0]
]

# 1. Writing to a CSV file
with open(dataset_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sample_data)
print("Data written successfully.")

# 2. Reading from a CSV file
with open(dataset_path, mode='r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        print(f"Row {i}:", row)

# %% [markdown]
# # Intermediate Examples

# %%
# 1. Using DictWriter to write dictionaries
dict_path = Path('dict_data.csv')
fieldnames = ['id', 'value', 'category']
data_dicts = [
    {'id': 1, 'value': 100, 'category': 'A'},
    {'id': 2, 'value': 200, 'category': 'B'}
]

with open(dict_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data_dicts)

# 2. Using DictReader to parse rows into dictionaries
print("\nReading with DictReader:")
with open(dict_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row['id'], row['category'])

# 3. Handling different delimiters (e.g., TSV)
tsv_path = Path('data.tsv')
with open(tsv_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows([['A', 'B'], ['1', '2']])

with open(tsv_path, mode='r') as file:
    reader = csv.reader(file, delimiter='\t')
    print("\nReading TSV:")
    for row in reader:
        print(row)

# %% [markdown]
# # Machine Learning Relevance
# Before training an ML model, you often need to stream massive datasets that don't fit into RAM. The `csv` module allows you to iterate over rows one by one, enabling you to extract features or filter out bad records incrementally. It's often used in customized PyTorch `Dataset` or TensorFlow `tf.data` pipelines.
# 
# # Common Mistakes
# - Forgetting `newline=''` when opening files for writing, leading to double-spaced rows on Windows.
# - Parsing numbers as strings; `csv` returns strings, so you must explicitly convert them to `float` or `int`.
# - Attempting to load a gigabyte-sized CSV entirely into memory with `list(csv.reader(f))` instead of iterating.
# 
# # Interview Questions
# 1. Why should you pass `newline=''` when writing CSV files in Python?
# 2. What is the advantage of `csv.DictReader` over `csv.reader`?
# 3. How do you handle commas inside a field when creating a CSV?
# 4. If you have a 10GB CSV file and 4GB of RAM, how would you process it?
# 5. How can you specify a custom delimiter, like a semicolon?
# 
# # Practice Problems
# 1. Write a script to convert the `sample_data.csv` to a pipe-separated file (`|`).
# 2. Read `dict_data.csv` and calculate the average of the 'value' column.
# 3. Write a function that takes a CSV file path and returns the number of rows (excluding the header).
# 4. Extract only the 'label' column from `sample_data.csv` into a Python list.
# 5. Delete the temporary files created in this lesson.
# 
# # Solutions

# %%
# Solution 1
with open(dataset_path, 'r') as f_in, open('pipe_data.csv', 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out, delimiter='|')
    writer.writerows(reader)

# Solution 2
with open(dict_path, 'r') as f:
    reader = csv.DictReader(f)
    values = [int(row['value']) for row in reader]
    print("Average value:", sum(values) / len(values) if values else 0)

# Solution 3
def count_rows(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        return sum(1 for row in reader)
print("Row count:", count_rows(dataset_path))

# Solution 4
with open(dataset_path, 'r') as f:
    reader = csv.DictReader(f) # Assumes first row is header
    labels = [row['label'] for row in reader]
    print("Labels:", labels)

# Solution 5
for p in [dataset_path, dict_path, tsv_path, Path('pipe_data.csv')]:
    if p.exists():
        p.unlink()

# %% [markdown]
# # Further Reading
# - [Official Python Documentation for csv](https://docs.python.org/3/library/csv.html)
