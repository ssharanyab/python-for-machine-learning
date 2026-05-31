# %% [markdown]
# # Title
# JSON: Serializing Nested Data
# 
# # Why this matters
# JavaScript Object Notation (JSON) is the universal format for configuration files, API responses, and metadata in machine learning projects. Hyperparameters, model architectures, and evaluation metrics are often logged in JSON format because it is both human-readable and easily parsed.
# 
# # Learning Objectives
# - Serialize (dump) Python dictionaries into JSON strings and files.
# - Deserialize (load) JSON data back into Python objects.
# - Format JSON for readability using indentation.
# - Handle common JSON serialization errors.
# 
# # Concept Explanation
# The `json` module provides a simple interface to convert Python dictionaries and lists into a text format that can be saved or transmitted over a network. The two main pairs of functions are:
# - `json.dumps()` / `json.loads()` for string manipulation.
# - `json.dump()` / `json.load()` for file I/O operations.
# 
# # Beginner Examples

# %%
import json

# 1. Dumping a dictionary to a JSON string
model_config = {
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "layers": [64, 128, 64],
    "activation": "relu"
}

json_string = json.dumps(model_config)
print("JSON String:", json_string)
print("Type:", type(json_string))

# 2. Loading a JSON string back to a dictionary
parsed_config = json.loads(json_string)
print("Parsed Config:", parsed_config["optimizer"])

# %% [markdown]
# # Intermediate Examples

# %%
from pathlib import Path

# 1. Writing JSON to a file with indentation for readability
config_path = Path("config.json")
with open(config_path, "w") as f:
    json.dump(model_config, f, indent=4)
print("Wrote formatted JSON to file.")

# 2. Reading JSON from a file
with open(config_path, "r") as f:
    loaded_config = json.load(f)
    print("Loaded layers:", loaded_config["layers"])

# 3. Handling un-serializable objects
import numpy as np

# A common error: NumPy arrays are not JSON serializable natively
bad_data = {"array": np.array([1, 2, 3])}
try:
    json.dumps(bad_data)
except TypeError as e:
    print("\nSerialization Error:", e)

# The fix: Convert to Python native lists
good_data = {"array": np.array([1, 2, 3]).tolist()}
print("Fixed Serialization:", json.dumps(good_data))

# %% [markdown]
# # Machine Learning Relevance
# You will frequently use JSON to save experiment tracking results. For example, after evaluating a model, you might save metrics like `{"accuracy": 0.95, "f1_score": 0.92}` to a `metrics.json` file. This allows visualization tools or hyperparameter tuning frameworks to easily ingest your results.
# 
# # Common Mistakes
# - Trying to serialize complex objects (like custom classes, NumPy arrays, or PyTorch tensors) directly. You must convert them to standard Python types (dicts, lists, int, float, str) first.
# - Confusing `json.dumps` (dump string) with `json.dump` (dump to file object).
# - Using single quotes for JSON strings; JSON requires double quotes, which the `json` module handles for you, but manually written JSON often has this error.
# 
# # Interview Questions
# 1. What is the difference between `json.dump` and `json.dumps`?
# 2. How do you format a JSON string to make it easily readable?
# 3. Can you serialize a Python tuple to JSON? What happens to it?
# 4. How would you handle a `TypeError` when attempting to dump a NumPy array to JSON?
# 5. Are JSON keys always strings? What happens if you use an integer as a key in Python before dumping?
# 
# # Practice Problems
# 1. Create a dictionary representing your ML model parameters and serialize it to a formatted JSON string.
# 2. Write the JSON string to a file named `params.json`.
# 3. Load the data from `params.json` and print the learning rate.
# 4. Create a dictionary with an integer key, serialize it, deserialize it, and check the type of the key.
# 5. Clean up by deleting `config.json` and `params.json`.
# 
# # Solutions

# %%
# Solution 1
params = {"batch_size": 32, "epochs": 100}
params_str = json.dumps(params, indent=2)
print("Formatted params:\n", params_str)

# Solution 2
params_path = Path("params.json")
with open(params_path, "w") as f:
    json.dump(params, f)

# Solution 3
with open(params_path, "r") as f:
    data = json.load(f)
    print("Batch size:", data["batch_size"])

# Solution 4
int_key_dict = {1: "one"}
serialized = json.dumps(int_key_dict)
deserialized = json.loads(serialized)
print("Original key type:", type(list(int_key_dict.keys())[0]))
print("Deserialized key type:", type(list(deserialized.keys())[0])) # Became a string!

# Solution 5
for p in [config_path, params_path]:
    if p.exists():
        p.unlink()

# %% [markdown]
# # Further Reading
# - [Official Python Documentation for json](https://docs.python.org/3/library/json.html)
