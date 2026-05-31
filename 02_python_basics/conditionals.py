# %% [markdown]
# # Conditionals
#
# # Why this matters
# Conditionals (`if`, `elif`, `else`) are the decision-making mechanisms in code. In Machine Learning, you use conditionals constantly: implementing early stopping, branching logic for different model architectures, filtering dirty data, or writing custom loss functions that penalize errors differently based on conditions.
#
# # Learning Objectives
# 1. Understand the syntax of `if`, `elif`, and `else` statements.
# 2. Learn how indentation defines code blocks in Python.
# 3. Master nested conditionals.
# 4. Use inline conditionals (ternary operators) for concise code.
#
# # Concept Explanation
# Conditional statements evaluate a boolean expression. If the expression is `True`, the indented block of code immediately beneath it executes. 
# - `if` initiates the conditional block.
# - `elif` (else if) checks another condition if previous conditions were False.
# - `else` provides a fallback block if all preceding conditions were False.
# Python heavily relies on whitespace (indentation) to denote scope, replacing the brackets `{}` used in C++ or Java.
#
# # Beginner Examples
# Let's create basic logic flows commonly seen in ML training scripts.

# %%
# Basic if-else
val_loss = 0.45
threshold = 0.5

if val_loss < threshold:
    print("Model performance is acceptable. Saving model...")
else:
    print("Model performance is poor. Continuing training...")

# if-elif-else chain
data_type = "image"

if data_type == "text":
    print("Loading NLP pipeline (Transformers/Tokenizers)")
elif data_type == "image":
    print("Loading Computer Vision pipeline (CNNs/Augmentations)")
elif data_type == "audio":
    print("Loading Audio pipeline (Spectrograms)")
else:
    print("Unknown data type.")

# %% [markdown]
# # Intermediate Examples
# Complex logic often requires nested conditions and ternary operators.

# %%
# Nested Conditionals (Early Stopping Logic)
current_epoch = 15
max_epochs = 50
loss_improved = False

if current_epoch > 10:
    if not loss_improved:
        print("Triggering Early Stopping: Loss hasn't improved after 10 epochs.")
    else:
        print("Training continuing normally.")
else:
    print("Too early to evaluate stopping criteria.")

# Inline conditional (Ternary Operator)
# Syntax: [value_if_true] if [condition] else [value_if_false]
is_gpu_available = True
device = "cuda" if is_gpu_available else "cpu"
print(f"Tensors will be allocated on: {device}")

# Truthiness
# Empty lists, None, 0, and empty strings evaluate to False.
missing_features = []
if not missing_features:
    print("All features are present. Proceeding with prediction.")

# %% [markdown]
# # Machine Learning Relevance
# A classic ML use case for conditionals is gradient clipping to prevent exploding gradients: `if gradient_norm > max_norm: scale_gradient()`. Another is handling missing values during data preprocessing: `if pd.isna(value): value = mean_imputation`. Mastering these flows ensures your pipelines are robust and don't crash when encountering edge-case data.
#
# # Common Mistakes
# 1. Indentation errors: mixing spaces and tabs, or inconsistent spaces, leading to `IndentationError`.
# 2. Forgetting the colon `:` at the end of the `if`, `elif`, or `else` statement.
# 3. Using `==` when comparing to `None`, `True`, or `False`. PEP 8 recommends using `is` and `is not` for singletons (e.g., `if var is None:` instead of `if var == None:`).
# 4. Deeply nested `if` statements (Spaghetti code) making it impossible to read. Refactor into functions or use early returns.
#
# # Interview Questions
# 1. **How does Python determine the scope of an `if` block?**
#    *Answer:* By indentation. All consecutive lines indented at the same level belong to the same block.
# 2. **What does the pass statement do?**
#    *Answer:* `pass` is a null operation. It's used as a placeholder in empty blocks (like an empty `if` or `function`) to prevent syntax errors.
# 3. **Explain "truthiness" in Python.**
#    *Answer:* Objects can be evaluated in a boolean context. By default, objects are true. Empty sequences (strings, lists, tuples), zero of any numeric type, `None`, and `False` are considered false.
# 4. **How do you write a ternary conditional in Python?**
#    *Answer:* `x = a if condition else b`.
# 5. **If you have multiple `elif` blocks, do they all get evaluated?**
#    *Answer:* No, Python stops evaluating as soon as it finds the first `if` or `elif` condition that is True.
#
# # Practice Problems
# 1. Write an `if-elif-else` statement that assigns a grade 'A' for score > 90, 'B' for > 80, 'C' for > 70, and 'F' otherwise.
# 2. Use a ternary operator to set `batch_size = 128` if `ram_gb > 16` else `batch_size = 32`.
# 3. Write a conditional that checks if a list `epochs_losses` is empty. If empty, print "No training data". If not, print the last loss value.
# 4. Implement a simple activation function (ReLU): Given an input `x`, if `x > 0`, return `x`, else return `0`.
# 5. Write a script that checks if a string `filename` ends with ".csv". If it does, print "Reading CSV", otherwise print "Unsupported format".
#
# # Solutions

# %%
# Solution to Problem 1
score = 85
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
else:
    grade = 'F'
print(f"Grade: {grade}")

# %%
# Solution to Problem 2
ram_gb = 8
batch_size = 128 if ram_gb > 16 else 32
print(f"Selected batch size: {batch_size}")

# %%
# Solution to Problem 3
epochs_losses = [0.8, 0.5, 0.3]
if not epochs_losses:
    print("No training data")
else:
    print(f"Latest loss: {epochs_losses[-1]}")

# %%
# Solution to Problem 4
x = -5
relu_output = x if x > 0 else 0
print(f"ReLU({x}) = {relu_output}")

# %%
# Solution to Problem 5
filename = "dataset.csv"
if filename.endswith(".csv"):
    print("Reading CSV")
else:
    print("Unsupported format")

# %% [markdown]
# # Further Reading
# - [Python Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
# - [Real Python - Conditional Statements](https://realpython.com/python-conditional-statements/)
