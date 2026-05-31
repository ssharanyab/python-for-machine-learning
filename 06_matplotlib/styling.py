# %% [markdown]
# # Styling
# 
# # Why this matters
# A beautifully styled plot is easier to read, more professional, and crucial for reports or presentations.
# 
# # Learning Objectives
# - Use built-in Matplotlib styles.
# - Customize colors, fonts, and line properties.
# - Work with color maps.
# 
# # Concept Explanation
# `matplotlib.style.use()` can apply sweeping visual themes. Fine-grained control is achieved via arguments like `color`, `linewidth`, and `fontdict`.
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np

# Using a built-in style
plt.style.use('ggplot')
x = np.linspace(0, 10, 50)

plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), linewidth=2, label="Sine")
plt.title("Styled Plot (ggplot)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# Custom colors and colormaps
plt.style.use('default') # Reset style
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100) # third variable for colormap

plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, c=z, cmap='coolwarm', s=100, edgecolor='black', label="Points")
plt.colorbar(sc, label="Z Value")

font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 16}
plt.title("Custom Colormap and Fonts", fontdict=font)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True, linestyle=':', color='gray')
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Presentations to stakeholders require clean, styled plots. Heatmaps (e.g. for confusion matrices) rely heavily on proper colormap choices (sequential vs diverging).
# 
# # Common Mistakes
# - Using styles but not resetting them (affecting all subsequent plots in a notebook).
# - Poor color choices that are not accessible to colorblind individuals.
# 
# # Interview Questions
# 1. How do you change the global style in Matplotlib?
# 2. What is a colormap, and when would you use a sequential vs diverging colormap?
# 3. How do you adjust line width and style?
# 4. How can you change font properties for titles and labels?
# 5. How do you reset matplotlib to its default style?
# 
# # Practice Problems
# 1. Plot a line chart using the 'seaborn-v0_8' or 'fivethirtyeight' style.
# 2. Plot a scatter plot using the 'plasma' colormap.
# 3. Create a plot and set the background color of the axes.
# 4. Change the tick colors and label sizes.
# 5. Save the styled plot with a transparent background.
# 
# # Solutions
# %%
# Solution 1
plt.style.use('fivethirtyeight')
plt.figure()
plt.plot([1,2,3], [1,4,9], label="Line")
plt.title("538 Style")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# Solution 2
plt.style.use('default')
plt.figure()
plt.scatter(np.random.rand(50), np.random.rand(50), c=np.random.rand(50), cmap='plasma', label="Plasma")
plt.title("Plasma Colormap")
plt.xlabel("X"); plt.ylabel("Y")
plt.colorbar()
plt.legend(); plt.grid(True)
plt.show()

# Solution 3
fig, ax = plt.subplots()
ax.set_facecolor('lightyellow')
ax.plot([1,2], [3,4], label="Line")
ax.set_title("Background Color")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.legend(); ax.grid(True)
plt.show()

# Solution 4
plt.figure()
plt.plot([1,2], [2,1], label="Line")
plt.tick_params(axis='x', colors='red', labelsize=14)
plt.tick_params(axis='y', colors='blue', labelsize=14)
plt.title("Custom Ticks")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# Solution 5
fig = plt.figure()
plt.plot([1,2], [1,2], label="Line")
plt.title("Transparent Save")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
# fig.savefig("transparent.png", transparent=True)
plt.show()

# %% [markdown]
# # Further Reading
# - Customizing Matplotlib with style sheets and rcParams
