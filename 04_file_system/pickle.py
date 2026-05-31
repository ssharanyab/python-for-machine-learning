# %% [markdown]
# # Title
# Pickle: Object Serialization
# 
# # Why this matters
# In machine learning, after spending hours or days training a model, you need a way to save it. `pickle` is Python's built-in mechanism for serializing and deserializing entire Python objects, allowing you to save trained models (like Scikit-Learn estimators), complex data structures, and preprocessed data arrays to disk.
# 
# # Learning Objectives
# - Serialize Python objects into byte streams using `pickle`.
# - Load serialized objects back into memory.
# - Understand the security implications of unpickling untrusted data.
# - Differentiate between Pickle and JSON for serialization.
# 
# # Concept Explanation
# Pickling converts a Python object hierarchy into a byte stream, while unpickling is the inverse operation. Unlike JSON, which is a cross-language standard for simple data, Pickle is Python-specific and can serialize complex objects, custom classes, and functions.
# 
# # Beginner Examples

# %%
import pickle

# 1. Pickling an object to a byte string
my_data = {'model_weights': [0.1, 0.5, -0.2], 'bias': 0.01}
pickled_bytes = pickle.dumps(my_data)
print("Pickled bytes:", pickled_bytes[:20], "...")

# 2. Unpickling bytes back to an object
restored_data = pickle.loads(pickled_bytes)
print("Restored data:", restored_data)

# %% [markdown]
# # Intermediate Examples

# %%
from pathlib import Path

# 1. Saving a mock ML model to a file
# Note the 'wb' mode for Writing Bytes
model_file = Path('mock_model.pkl')
mock_model = {"coef_": [1.5, 2.5], "intercept_": -0.5, "type": "LinearRegression"}

with open(model_file, 'wb') as f:
    pickle.dump(mock_model, f)
print("Model saved to disk.")

# 2. Loading the model from a file
# Note the 'rb' mode for Reading Bytes
with open(model_file, 'rb') as f:
    loaded_model = pickle.load(f)
print("Loaded Model Intercept:", loaded_model["intercept_"])

# 3. Handling custom classes
class Preprocessor:
    def __init__(self, scale):
        self.scale = scale
    def process(self, x):
        return x * self.scale

p = Preprocessor(0.5)
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(p, f)

with open('preprocessor.pkl', 'rb') as f:
    restored_p = pickle.load(f)
    print("Processed value:", restored_p.process(10))

# %% [markdown]
# # Machine Learning Relevance
# Libraries like `scikit-learn` use `pickle` (or its optimized cousin, `joblib`) heavily for saving models (`model.pkl`). This makes model deployment possible, as you can train a model locally, pickle it, and unpickle it on a cloud server to serve predictions.
# 
# # Common Mistakes
# - **SECURITY WARNING**: Never unpickle data from an untrusted source! Pickle can execute arbitrary code during unpickling. Use JSON for receiving data over an API.
# - Forgetting to use binary mode (`'wb'` or `'rb'`) when opening files for pickling.
# - Pickling custom objects and later trying to unpickle them without the class definition being present in the environment.
# 
# # Interview Questions
# 1. What is the main difference between JSON and Pickle?
# 2. Why is it dangerous to unpickle data from untrusted sources?
# 3. What file mode must be used when saving a pickled object to a file?
# 4. Can Pickle serialize functions and custom classes?
# 5. What happens if you modify a class definition after pickling an instance, and then try to unpickle it?
# 
# # Practice Problems
# 1. Create a dictionary representing model hyperparameters and pickle it to a file named `hyper.pkl`.
# 2. Read `hyper.pkl` and print the dictionary.
# 3. Explain in a comment why you wouldn't use Pickle for a web API response.
# 4. Attempt to pickle a lambda function and observe the error.
# 5. Clean up by deleting the `.pkl` files created.
# 
# # Solutions

# %%
# Solution 1
hyperparams = {"alpha": 0.01, "max_iter": 1000}
with open('hyper.pkl', 'wb') as f:
    pickle.dump(hyperparams, f)

# Solution 2
with open('hyper.pkl', 'rb') as f:
    data = pickle.load(f)
    print("Loaded hyperparams:", data)

# Solution 3
# You wouldn't use Pickle for a web API because it's Python-specific (other languages can't read it easily)
# and it poses a massive security risk if the client sends malicious pickled data.

# Solution 4
my_lambda = lambda x: x * 2
try:
    pickle.dumps(my_lambda)
except Exception as e:
    print("Lambda pickling error:", type(e).__name__)

# Solution 5
for p in [model_file, Path('preprocessor.pkl'), Path('hyper.pkl')]:
    if p.exists():
        p.unlink()

# %% [markdown]
# # Further Reading
# - [Official Python Documentation for pickle](https://docs.python.org/3/library/pickle.html)
