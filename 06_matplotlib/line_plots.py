# %% [markdown]
# # Line Plots
# 
# # Why this matters
# Line plots are the fundamental way to visualize trends over time or continuous intervals in data, making them essential for time series forecasting and understanding continuous relationships.
# 
# # Learning Objectives
# - Understand how to create basic line plots using matplotlib.
# - Learn to customize line styles, colors, and markers.
# - Be able to plot multiple lines on the same axes.
# - Add titles, axis labels, legends, and grids to plots.
# 
# # Concept Explanation
# A line plot connects individual data points with line segments. In `matplotlib`, `plt.plot()` is used for this purpose. It is typically used for ordered data, like time series.
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Basic line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Linear Trend", color="blue", marker="o")
plt.title("Basic Line Plot")
plt.xlabel("X-axis (Time)")
plt.ylabel("Y-axis (Value)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# Multiple lines with styles
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y1, label="Sine Wave", color="red", linestyle="--")
plt.plot(x, y2, label="Cosine Wave", color="green", linestyle="-.")
plt.title("Trigonometric Functions over Time")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Time series data mock
dates = pd.date_range('20230101', periods=12, freq='ME')
sales = np.random.randint(100, 500, size=12)
plt.figure(figsize=(10, 5))
plt.plot(dates, sales, label="Monthly Sales", marker='s')
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Line plots are crucial in Machine Learning for plotting learning curves (e.g., training vs. validation loss over epochs) and evaluating models over time.
# 
# # Common Mistakes
# - Forgetting to call `plt.show()`.
# - Not adding labels and legends, making the plot uninterpretable.
# - Plotting categorical data on the x-axis with `plt.plot()` instead of a bar chart.
# 
# # Interview Questions
# 1. How do you plot multiple lines on the same graph in matplotlib?
# 2. What is the difference between `plot()` and `scatter()`?
# 3. How do you adjust the line style and color in `plt.plot()`?
# 4. Why might you use a logarithmic scale on a line plot, and how is it done?
# 5. How do you save a matplotlib plot to an image file?
# 
# # Practice Problems
# 1. Plot $y = x^2$ for x from -10 to 10 with a red dashed line.
# 2. Plot two lines, one for train accuracy and one for val accuracy over 10 epochs.
# 3. Create a line plot with a logarithmic y-axis for exponentially growing data.
# 4. Plot a line graph where the markers are stars.
# 5. Load a mock dataset `../datasets/sales.csv` (or use pandas dummy data) and plot the trend over a column 'Date'.
# 
# # Solutions
# %%
# Solution 1
x = np.linspace(-10, 10, 100)
y = x**2
plt.figure()
plt.plot(x, y, 'r--', label='y = x^2')
plt.title("Quadratic Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Solution 2
epochs = np.arange(1, 11)
train_acc = np.linspace(0.5, 0.95, 10)
val_acc = np.linspace(0.4, 0.85, 10)
plt.figure()
plt.plot(epochs, train_acc, label="Train", marker="o")
plt.plot(epochs, val_acc, label="Validation", marker="x")
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Solution 3
x = np.arange(1, 10)
y = np.exp(x)
plt.figure()
plt.plot(x, y, label="Exponential Growth")
plt.yscale('log')
plt.title("Log Scale Plot")
plt.xlabel("X")
plt.ylabel("Y (Log Scale)")
plt.legend()
plt.grid(True)
plt.show()

# Solution 4
plt.figure()
plt.plot([1,2,3], [4,5,2], marker="*", markersize=10, label="Star markers")
plt.title("Star Markers Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Solution 5
dates = pd.date_range('2023-01-01', periods=5)
sales = [120, 150, 130, 180, 200]
df = pd.DataFrame({'Date': dates, 'Sales': sales})
plt.figure()
plt.plot(df['Date'], df['Sales'], label="Sales Trend")
plt.title("Sales over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Further Reading
# - Matplotlib documentation on Pyplot
# - Time Series Analysis with Python
