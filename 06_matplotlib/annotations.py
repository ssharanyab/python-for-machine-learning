# %% [markdown]
# # Annotations
# 
# # Why this matters
# Annotations guide the viewer's attention to key findings, outliers, or important milestones directly on the plot.
# 
# # Learning Objectives
# - Add text to plots using `plt.text()`.
# - Point out specific data points using `plt.annotate()` with arrows.
# - Use horizontal and vertical lines to highlight thresholds.
# 
# # Concept Explanation
# `plt.text(x, y, string)` places text at data coordinates. `plt.annotate()` can place text and an arrow pointing to a specific point.
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np

# Basic Text
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Sine")
plt.text(1.57, 1.05, "Local Max", fontsize=12, color='red', ha='center')
plt.title("Plot with Text")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# Annotation with Arrows and Threshold lines
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Sine")

# Arrow annotation
plt.annotate('Minimum', xy=(4.71, -1), xytext=(5.5, -0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Threshold lines
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(np.pi, color='green', linestyle=':', label="Pi Threshold")

plt.title("Annotations and Thresholds")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Used to annotate points of interest in learning curves (e.g., "Early Stopping Point" or "Overfitting Starts") or threshold boundaries on ROC curves.
# 
# # Common Mistakes
# - Hardcoding text coordinates incorrectly when data scales change.
# - Cluttering the plot with too many text labels.
# 
# # Interview Questions
# 1. How do you add a horizontal line across the entire plot axis?
# 2. What is the difference between `text()` and `annotate()`?
# 3. How do you format the arrow in `plt.annotate()`?
# 4. How can you align text (e.g., center-aligned) using `plt.text()`?
# 5. How do you highlight a shaded region on a plot?
# 
# # Practice Problems
# 1. Add text to a plot at data coordinates (2, 5).
# 2. Annotate the maximum point of a quadratic curve with an arrow.
# 3. Draw a red dashed vertical line at x=5.
# 4. Highlight the area between x=2 and x=4 using `plt.axvspan()`.
# 5. Add a bounding box to your text using the `bbox` parameter.
# 
# # Solutions
# %%
# Solution 1
plt.figure()
plt.plot([0, 5], [0, 10], label="Line")
plt.text(2, 5, "Point (2,5)", color="blue")
plt.title("Text Annotation")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# Solution 2
x = np.linspace(-5, 5, 50)
y = -x**2
plt.figure()
plt.plot(x, y, label="Curve")
plt.annotate('Global Max', xy=(0, 0), xytext=(2, -10),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
plt.title("Arrow Annotation")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# Solution 3
plt.figure()
plt.plot([0, 10], [0, 10], label="Data")
plt.axvline(5, color='red', linestyle='--', label="Threshold")
plt.title("Vertical Line")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# Solution 4
plt.figure()
plt.plot([0, 10], [0, 10], label="Data")
plt.axvspan(2, 4, color='yellow', alpha=0.3, label="Region")
plt.title("Shaded Region")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# Solution 5
plt.figure()
plt.plot([0, 10], [0, 10], label="Data")
plt.text(5, 5, "Important Text", bbox=dict(facecolor='white', edgecolor='black'))
plt.title("Text with Bounding Box")
plt.xlabel("X"); plt.ylabel("Y")
plt.legend(); plt.grid(True)
plt.show()

# %% [markdown]
# # Further Reading
# - Text rendering with Matplotlib
