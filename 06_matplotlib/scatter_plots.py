# %% [markdown]
# # Scatter Plots
# 
# # Why this matters
# Scatter plots are essential for identifying relationships, correlations, and clusters between two continuous variables.
# 
# # Learning Objectives
# - Understand how to create scatter plots using `plt.scatter()`.
# - Map third variables to color (c) or size (s).
# - Customize marker styles and opacities.
# 
# # Concept Explanation
# A scatter plot places points on a Cartesian coordinate system. It is ideal for observing how much one variable is affected by another (correlation).
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np

# Basic scatter plot
np.random.seed(42)
x = np.random.rand(50)
y = 2 * x + np.random.normal(0, 0.1, 50)

plt.figure(figsize=(8, 4))
plt.scatter(x, y, label="Data Points", color="purple")
plt.title("Basic Scatter Plot")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# Scatter plot with size and color mapping
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis', label="Random Bubble")
plt.colorbar(scatter, label="Color Intensity")
plt.title("Bubble Chart (Scatter with Size and Color)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Scatter plots are heavily used to visualize feature relationships (e.g., pairplots), evaluate regression model predictions (Predicted vs Actual), and plot clustering results (e.g., K-Means).
# 
# # Common Mistakes
# - Using a scatter plot for sequential data (where a line plot is better).
# - Overplotting: too many points making the plot dense (fix with `alpha` parameter).
# 
# # Interview Questions
# 1. How do you change the color of points based on a third categorical variable?
# 2. What does the `alpha` parameter do in `plt.scatter()`?
# 3. How do you add a colorbar to a scatter plot?
# 4. How would you visualize predicted vs actual values in a regression model?
# 5. What are the advantages of using `scatter()` over `plot(marker='o')`?
# 
# # Practice Problems
# 1. Create a scatter plot of 100 random points.
# 2. Change the marker style to triangles.
# 3. Create a scatter plot where the color of points changes based on their x-value.
# 4. Plot two different clusters of points in two different colors.
# 5. Add a grid, title, and legend to a scatter plot showing height vs weight.
# 
# # Solutions
# %%
# Solution 1
plt.figure()
plt.scatter(np.random.rand(100), np.random.rand(100), label="Random")
plt.title("100 Random Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Solution 2
plt.figure()
plt.scatter(np.random.rand(20), np.random.rand(20), marker='^', label="Triangles")
plt.title("Triangle Markers")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Solution 3
x = np.random.rand(50)
y = np.random.rand(50)
plt.figure()
plt.scatter(x, y, c=x, cmap='plasma', label="Mapped Color")
plt.title("Color mapped to X")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Solution 4
x1, y1 = np.random.normal(2, 0.5, 50), np.random.normal(2, 0.5, 50)
x2, y2 = np.random.normal(5, 0.5, 50), np.random.normal(5, 0.5, 50)
plt.figure()
plt.scatter(x1, y1, color='red', label="Cluster 1")
plt.scatter(x2, y2, color='blue', label="Cluster 2")
plt.title("Clusters")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Solution 5
heights = np.random.normal(170, 10, 100)
weights = heights * 0.4 + np.random.normal(0, 5, 100)
plt.figure()
plt.scatter(heights, weights, alpha=0.6, label="Individuals")
plt.title("Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Further Reading
# - Matplotlib Scatter Documentation
# - Visualizing Multidimensional Data
