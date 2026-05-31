# %% [markdown]
# # Bar Charts
# 
# # Why this matters
# Bar charts are the primary way to visualize categorical data and compare quantities across different groups.
# 
# # Learning Objectives
# - Create vertical and horizontal bar charts.
# - Plot stacked and grouped bar charts.
# - Customize bar colors, widths, and edge colors.
# 
# # Concept Explanation
# Bar charts represent categories as bars, with the height (or length) proportional to the values they represent.
# 
# # Beginner Examples
# %%
import matplotlib.pyplot as plt
import numpy as np

# Basic Bar Chart
categories = ['Apples', 'Bananas', 'Cherries', 'Dates']
values = [25, 40, 15, 30]

plt.figure(figsize=(8, 4))
plt.bar(categories, values, color='coral', edgecolor='black', label="Fruit Sales")
plt.title("Basic Bar Chart")
plt.xlabel("Fruits")
plt.ylabel("Quantity Sold")
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.show()

# %% [markdown]
# # Intermediate Examples
# %%
# Horizontal and Grouped Bar Charts
# Horizontal
plt.figure(figsize=(8, 4))
plt.barh(categories, values, color='teal', label="Fruit Sales")
plt.title("Horizontal Bar Chart")
plt.xlabel("Quantity Sold")
plt.ylabel("Fruits")
plt.legend()
plt.grid(axis='x', linestyle='--')
plt.show()

# Grouped
x = np.arange(len(categories))
width = 0.35
sales_2022 = [20, 35, 10, 25]
sales_2023 = [25, 40, 15, 30]

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, sales_2022, width, label='2022', color='lightblue')
plt.bar(x + width/2, sales_2023, width, label='2023', color='darkblue')

plt.title("Grouped Bar Chart")
plt.xlabel("Fruits")
plt.ylabel("Sales")
plt.xticks(x, categories)
plt.legend()
plt.grid(axis='y')
plt.show()

# %% [markdown]
# # Machine Learning Relevance
# Used for visualizing class imbalances in classification tasks or feature importance scores extracted from models like Random Forest.
# 
# # Common Mistakes
# - Using line plots for categorical data instead of bar charts.
# - Overcomplicating charts with 3D effects.
# 
# # Interview Questions
# 1. How do you create a grouped bar chart in matplotlib?
# 2. How can you plot horizontal bars instead of vertical?
# 3. What is a stacked bar chart and how do you plot it?
# 4. How do you change the width of the bars?
# 5. How can you add error bars to a bar chart?
# 
# # Practice Problems
# 1. Create a bar chart showing the count of 3 different categories.
# 2. Add error bars to a bar chart.
# 3. Create a stacked bar chart.
# 4. Change the colors of individual bars based on their value.
# 5. Rotate the x-axis labels by 45 degrees.
# 
# # Solutions
# %%
# Solution 1
plt.figure()
plt.bar(['A', 'B', 'C'], [10, 20, 15], label="Categories")
plt.title("Category Counts")
plt.xlabel("Category")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()

# Solution 2
plt.figure()
plt.bar(['A', 'B', 'C'], [10, 20, 15], yerr=[1, 2, 1.5], capsize=5, label="With Error")
plt.title("Bar Chart with Error Bars")
plt.xlabel("Category")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Solution 3
cats = ['Cat1', 'Cat2']
v1 = [10, 15]
v2 = [5, 10]
plt.figure()
plt.bar(cats, v1, label="Part 1")
plt.bar(cats, v2, bottom=v1, label="Part 2")
plt.title("Stacked Bar Chart")
plt.xlabel("Category")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Solution 4
values = [5, 15, 10]
colors = ['red' if v < 10 else 'green' for v in values]
plt.figure()
plt.bar(['A', 'B', 'C'], values, color=colors, label="Value Based Color")
plt.title("Conditional Colors")
plt.xlabel("Category")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Solution 5
plt.figure()
plt.bar(['Very Long Label 1', 'Very Long Label 2'], [10, 20], label="Rotated")
plt.xticks(rotation=45)
plt.title("Rotated Labels")
plt.xlabel("Category")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Further Reading
# - Matplotlib Bar Chart examples
