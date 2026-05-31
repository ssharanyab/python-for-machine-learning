# %% [markdown]
# # Histograms
# 
# # Why this matters
# Histograms are critical for understanding the distribution of a continuous variable. They show frequencies of data falling into different bins.
# 
# # Learning Objectives
# - Create basic histograms using `plt.hist()`.
# - Understand and manipulate bin sizes.
# - Plot multiple distributions on the same graph.
# 
# # Concept Explanation
# A histogram groups continuous data into bins (intervals) and plots the count (or density) of items in each bin as a bar.
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np

# Basic histogram
data = np.random.randn(1000)

plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', label="Normal Dist")
plt.title("Basic Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# Multiple histograms and density
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(3, 1.5, 1000)

plt.figure(figsize=(10, 5))
plt.hist(data1, bins=30, alpha=0.5, label='Distribution 1', density=True)
plt.hist(data2, bins=30, alpha=0.5, label='Distribution 2', density=True)
plt.title("Overlaid Density Histograms")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Used for Exploratory Data Analysis (EDA) to check if features follow a normal distribution, detect outliers, and decide on transformations (like log scale) before feeding to models.
# 
# # Common Mistakes
# - Choosing too few or too many bins, which can obscure the underlying distribution.
# - Confusing histograms with bar charts (histograms are for continuous data, bar charts for categorical).
# 
# # Interview Questions
# 1. What is the purpose of the `bins` parameter?
# 2. How does `density=True` change the y-axis?
# 3. How do you overlay two histograms transparently?
# 4. What is the difference between a histogram and a bar chart?
# 5. How can you add edge colors to the bins for better visibility?
# 
# # Practice Problems
# 1. Plot a histogram for a uniform distribution.
# 2. Plot a histogram with exactly 10 bins.
# 3. Create a cumulative histogram.
# 4. Plot two histograms side-by-side (using `hist` with a list of arrays).
# 5. Create a histogram with horizontal orientation.
# 
# # Solutions
# %%
# Solution 1
plt.figure()
plt.hist(np.random.uniform(0, 10, 1000), bins=20, label="Uniform")
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Solution 2
plt.figure()
plt.hist(np.random.randn(500), bins=10, edgecolor='black', label="10 Bins")
plt.title("10-Bin Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Solution 3
plt.figure()
plt.hist(np.random.randn(500), bins=30, cumulative=True, label="Cumulative")
plt.title("Cumulative Histogram")
plt.xlabel("Value")
plt.ylabel("Cumulative Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Solution 4
d1 = np.random.randn(500)
d2 = np.random.randn(500) + 2
plt.figure()
plt.hist([d1, d2], bins=15, label=['Dist 1', 'Dist 2'])
plt.title("Side-by-side Histograms")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# Solution 5
plt.figure()
plt.hist(np.random.randn(500), bins=20, orientation='horizontal', label="Horizontal")
plt.title("Horizontal Histogram")
plt.xlabel("Frequency")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Further Reading
# - Matplotlib Hist Documentation
# - Kernel Density Estimation (KDE) with Seaborn
