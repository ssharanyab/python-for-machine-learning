# %% [markdown]
# # Subplots
# 
# # Why this matters
# Complex analysis requires visualizing multiple related plots side-by-side. Subplots allow grid-like layouts for plots.
# 
# # Learning Objectives
# - Use `plt.subplots()` to create a grid of plots.
# - Understand the difference between Figure and Axes objects.
# - Share axes among subplots.
# 
# # Concept Explanation
# A `Figure` is the overall window or page. `Axes` are the individual plots. `plt.subplots(nrows, ncols)` creates a figure and a set of subplots.
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np

# Basic Subplots (1x2 grid)
x = np.linspace(0, 5, 50)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(x, x**2, color='red', label='Quadratic')
ax1.set_title("Square")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.grid(True)

ax2.plot(x, x**3, color='blue', label='Cubic')
ax2.set_title("Cube")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# 2x2 grid with shared axes
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

axs[0, 0].scatter(x, x + np.random.rand(50), label="Scatter")
axs[0, 0].set_title("Subplot 1")

axs[0, 1].plot(x, np.sin(x), label="Sine")
axs[0, 1].set_title("Subplot 2")

axs[1, 0].hist(np.random.randn(100), bins=20, label="Hist")
axs[1, 0].set_title("Subplot 3")

axs[1, 1].bar(['A', 'B'], [5, 10], label="Bar")
axs[1, 1].set_title("Subplot 4")

for ax in axs.flat:
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Perfect for side-by-side comparisons like Training vs Validation loss, or showing images from a dataset in a grid layout.
# 
# # Common Mistakes
# - Forgetting `plt.tight_layout()`, causing overlapping labels.
# - Confusing the syntax for `plt.` commands vs `Axes` methods (e.g., `plt.title()` vs `ax.set_title()`).
# 
# # Interview Questions
# 1. What is the difference between a Figure and an Axes in Matplotlib?
# 2. How do you share the x-axis across multiple subplots?
# 3. What does `plt.tight_layout()` do?
# 4. How can you iterate over a 2D array of Axes objects?
# 5. How do you create an irregularly sized subplot (e.g. one plot spanning two columns)?
# 
# # Practice Problems
# 1. Create a 3x1 grid of line plots.
# 2. Create a 1x2 grid where the second plot shares the y-axis with the first.
# 3. Set a common title for the entire figure using `fig.suptitle()`.
# 4. Iterate over a 2x2 grid to plot 4 different random histograms.
# 5. Use `GridSpec` or `subplot2grid` to make a large plot on top and two smaller ones below.
# 
# # Solutions
# %%
# Solution 1
fig, axs = plt.subplots(3, 1, figsize=(6, 8))
for i in range(3):
    axs[i].plot([1,2], [i, i*2], label=f"Line {i}")
    axs[i].set_title(f"Plot {i}")
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    axs[i].grid(True)
    axs[i].legend()
plt.tight_layout()
plt.show()

# Solution 2
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot([1,2], [3,4], label="P1")
ax1.set_title("P1")
ax1.grid(True); ax1.legend()
ax2.plot([1,2], [4,5], label="P2")
ax2.set_title("P2")
ax2.grid(True); ax2.legend()
ax1.set_xlabel("X"); ax1.set_ylabel("Y")
ax2.set_xlabel("X")
plt.show()

# Solution 3
fig, ax = plt.subplots()
ax.plot([1,2], [1,2], label="Line")
ax.grid(True); ax.legend()
ax.set_xlabel("X"); ax.set_ylabel("Y")
fig.suptitle("Main Figure Title")
plt.show()

# Solution 4
fig, axs = plt.subplots(2, 2)
for ax in axs.flat:
    ax.hist(np.random.randn(100), label="Dist")
    ax.set_title("Hist")
    ax.grid(True); ax.legend()
    ax.set_xlabel("X"); ax.set_ylabel("Y")
plt.tight_layout()
plt.show()

# Solution 5
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

for ax, name in zip([ax1, ax2, ax3], ["Top", "Bottom Left", "Bottom Right"]):
    ax.plot([1,2], label=name)
    ax.set_title(name)
    ax.grid(True); ax.legend()
    ax.set_xlabel("X"); ax.set_ylabel("Y")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Further Reading
# - Object-oriented Matplotlib Guide
