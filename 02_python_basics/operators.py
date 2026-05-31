# %% [markdown]
# # Operators
#
# # Why this matters
# Machine learning algorithms are fundamentally built on mathematical operations and logical comparisons. Understanding Python's operators enables you to write equations, manipulate matrices, threshold data, and construct conditional logic for model training loops.
#
# # Learning Objectives
# 1. Master arithmetic operators for mathematical calculations.
# 2. Utilize comparison operators for data filtering and evaluation.
# 3. Apply logical operators to combine conditions.
# 4. Understand assignment and bitwise operators.
#
# # Concept Explanation
# Operators are special symbols that carry out arithmetic or logical computation.
# - **Arithmetic:** `+`, `-`, `*`, `/` (float division), `//` (floor division), `%` (modulo), `**` (exponentiation).
# - **Comparison:** `==`, `!=`, `>`, `<`, `>=`, `<=`. Return booleans.
# - **Logical:** `and`, `or`, `not`. Used to combine boolean expressions.
# - **Assignment:** `=`, `+=`, `-=`, etc.
# - **Identity / Membership:** `is`, `is not`, `in`, `not in`.
#
# # Beginner Examples
# Let's perform basic math and comparisons, often used when evaluating loss or accuracy.

# %%
# Arithmetic
true_positives = 85
false_positives = 15

total_predictions = true_positives + false_positives
precision = true_positives / total_predictions
print(f"Precision: {precision}")

# Floor division and Modulo (useful for batching)
total_samples = 105
batch_size = 32
num_full_batches = total_samples // batch_size
remainder_samples = total_samples % batch_size
print(f"Batches: {num_full_batches}, Remainder: {remainder_samples}")

# Exponentiation (e.g., squared error)
error = -4
squared_error = error ** 2
print(f"Squared Error: {squared_error}")

# %% [markdown]
# # Intermediate Examples
# Here we combine operators to simulate logic used in data filtering and hyperparameter scheduling.

# %%
# Comparison and Logical Operators
accuracy = 0.92
loss = 0.15

# Check if model meets deployment criteria
is_good_model = (accuracy > 0.90) and (loss < 0.20)
print(f"Ready for deployment? {is_good_model}")

# Membership operators
valid_optimizers = ["adam", "sgd", "rmsprop"]
chosen_opt = "adamax"

if chosen_opt not in valid_optimizers:
    print(f"Warning: {chosen_opt} is not in standard list.")

# Assignment Operators in a loop
epoch_loss = 100
decay_rate = 0.9
print(f"Initial loss: {epoch_loss}")
epoch_loss *= decay_rate # Equivalent to: epoch_loss = epoch_loss * decay_rate
print(f"Loss after decay: {epoch_loss}")

# %% [markdown]
# # Machine Learning Relevance
# In Deep Learning, you constantly compute gradients, update weights (`w -= learning_rate * grad`), and evaluate metrics (`correct = preds == labels`). Python's base operators are overloaded by libraries like NumPy and PyTorch so that `+` adds entire matrices element-wise. Understanding floor division `//` is critical when calculating convolution output dimensions or splitting datasets into batches.
#
# # Common Mistakes
# 1. Using `/` when you need integer division, causing errors in functions that expect integer array indices. Use `//`.
# 2. Confusing logical operators (`and`, `or`) with bitwise operators (`&`, `|`). In Pandas/NumPy, you MUST use `&` and `|` for element-wise array comparisons, not `and`/`or`.
# 3. Forgetting operator precedence (e.g., `not A == B` vs `not (A == B)`). Use parentheses to be explicit.
# 4. Using assignment `=` instead of comparison `==` in if-statements (though Python 3.8+ introduced the walrus operator `:=` which blurs this line, standard `=` is invalid in an `if` expression).
#
# # Interview Questions
# 1. **What is the difference between `/` and `//` in Python?**
#    *Answer:* `/` always returns a float (true division). `//` performs floor division, rounding down to the nearest integer.
# 2. **In Pandas or NumPy, why do you get an error if you try to filter a DataFrame using `df[(df['age'] > 20) and (df['age'] < 30)]`?**
#    *Answer:* The Python `and` operator tries to evaluate the truth value of the entire Series, which is ambiguous. You must use the bitwise `&` operator for element-wise boolean logic: `(df['age'] > 20) & (df['age'] < 30)`.
# 3. **What is the result of `3 ** 2 ** 3`?**
#    *Answer:* Exponentiation is right-associative. It evaluates as `3 ** (2 ** 3)` = `3 ** 8` = 6561.
# 4. **Explain the `in` operator.**
#    *Answer:* It's a membership operator used to test if a sequence (like a list, tuple, or string) contains a specific value.
# 5. **What does the modulo operator `%` do and what's a common use case in ML training loops?**
#    *Answer:* It returns the remainder of a division. It's often used to trigger an action every N steps (e.g., `if step % 100 == 0: print(loss)`).
#
# # Practice Problems
# 1. Given `total_images = 543` and `batch_size = 64`, calculate the number of full batches and the number of images in the final partial batch.
# 2. Write a boolean expression that evaluates to True if a variable `p` is between 0 and 1 (inclusive).
# 3. Using the assignment operator `+=`, write a loop that calculates the sum of numbers from 1 to 5.
# 4. Given a predicted label `y_pred = 1` and actual label `y_true = 0`, compute the absolute error using arithmetic operators.
# 5. Test if the substring "deep" is present in the string "deep_learning_model".
#
# # Solutions

# %%
# Solution to Problem 1
total_images = 543
batch_size = 64
full_batches = total_images // batch_size
partial_batch = total_images % batch_size
print(f"Full batches: {full_batches}, Partial batch images: {partial_batch}")

# %%
# Solution to Problem 2
p = 0.85
is_valid_prob = 0 <= p <= 1 # Python supports chained comparisons!
print(f"Is {p} a valid probability? {is_valid_prob}")

# %%
# Solution to Problem 3
total_sum = 0
for i in [1, 2, 3, 4, 5]:
    total_sum += i
print(f"Sum: {total_sum}")

# %%
# Solution to Problem 4
y_pred = 1
y_true = 0
error = abs(y_true - y_pred) # or (y_true - y_pred) ** 2 for squared error
print(f"Absolute Error: {error}")

# %%
# Solution to Problem 5
string_name = "deep_learning_model"
has_deep = "deep" in string_name
print(f"Contains 'deep'? {has_deep}")

# %% [markdown]
# # Further Reading
# - [Python Basic Operators](https://www.w3schools.com/python/python_operators.asp)
# - [Operator Precedence in Python](https://docs.python.org/3/reference/expressions.html#operator-precedence)
