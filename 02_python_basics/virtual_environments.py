# %% [markdown]
# # Virtual Environments
#
# ## Why this matters
# In machine learning, projects rely on specific versions of libraries (e.g., TensorFlow 2.10, NumPy 1.23). If you install everything globally, updating a library for Project A might break Project B. Virtual environments isolate your project's dependencies, ensuring reproducibility and stability.
#
# ## Learning Objectives
# - Understand the concept and importance of virtual environments.
# - Learn how to create, activate, and deactivate a virtual environment using `venv`.
# - Learn how to manage dependencies using `pip` and `requirements.txt`.
#
# ## Concept Explanation
# A virtual environment is an isolated directory tree that contains its own Python installation and its own set of installed libraries.
# Common tools:
# - `venv`: Built into Python 3.
# - `conda`: Popular in data science, manages Python versions as well as libraries.
#
# ## Beginner Examples

# %% [markdown]
# ### Example 1: Creating a Virtual Environment
# Open your terminal and navigate to your project directory. Run:
# ```bash
# python -m venv myenv
# ```
# This creates a folder named `myenv` containing the isolated Python environment.

# %% [markdown]
# ### Example 2: Activating the Environment
# You must activate the environment for your terminal session to use it.
# - **Windows (Command Prompt):** `myenv\Scripts\activate.bat`
# - **Windows (PowerShell):** `myenv\Scripts\Activate.ps1`
# - **macOS/Linux:** `source myenv/bin/activate`
#
# Once activated, your command prompt will prefix the environment name: `(myenv) $`

# %% [markdown]
# ### Example 3: Deactivating
# To leave the virtual environment and return to the global Python, simply type:
# ```bash
# deactivate
# ```

# %% [markdown]
# ## Intermediate Examples

# %% [markdown]
# ### Example 4: Installing Packages
# While the environment is activated, use `pip` to install packages:
# ```bash
# pip install numpy pandas scikit-learn
# ```
# These are installed locally in `myenv/`, not globally.

# %% [markdown]
# ### Example 5: Saving Dependencies (requirements.txt)
# To share your project, you need a list of dependencies. Generate it with:
# ```bash
# pip freeze > requirements.txt
# ```
# This file looks like:
# ```text
# numpy==1.23.5
# pandas==1.5.3
# ```

# %% [markdown]
# ### Example 6: Replicating an Environment
# When a colleague clones your repo, they create a new virtual environment and install the exact versions from your file:
# ```bash
# pip install -r requirements.txt
# ```

# %% [markdown]
# ## Machine Learning Relevance
# Reproducibility is a core tenet of data science. Without virtual environments and a `requirements.txt` (or `environment.yml` for conda), you cannot guarantee that your model will train or evaluate identically on another machine.

# %% [markdown]
# ## Common Mistakes
# 1. **Committing the `venv` folder to Git:** Virtual environment folders contain thousands of files and binaries specific to your OS. Always add your `venv` folder name (e.g., `myenv/`) to your `.gitignore` file.
# 2. **Forgetting to Activate:** Installing pip packages globally because you forgot to run the activation script.
# 3. **Not Freezing Dependencies:** Sharing a Jupyter Notebook without the `requirements.txt`, leaving the user guessing which versions of libraries are required.

# %% [markdown]
# ## Interview Questions
# 1. Why do we need virtual environments in Python?
# 2. What is the difference between `venv` and `conda`?
# 3. How do you generate and use a `requirements.txt` file?
# 4. What should you do with your virtual environment directory in relation to Git?
# 5. How do you deactivate an active virtual environment?

# %% [markdown]
# ## Practice Problems
# 1. Open your terminal, navigate to a safe folder, and create a virtual environment named `ml_env`.
# 2. Activate the `ml_env` virtual environment.
# 3. Install the library `requests` inside the virtual environment.
# 4. Generate a `requirements.txt` file.
# 5. Deactivate the environment.
#
# *(Note: These are terminal exercises, no Python code is required below)*

# %% [markdown]
# ## Solutions
# ```bash
# # Solution 1
# python -m venv ml_env
#
# # Solution 2 (Windows PowerShell)
# .\ml_env\Scripts\Activate.ps1
#
# # Solution 3
# pip install requests
#
# # Solution 4
# pip freeze > requirements.txt
#
# # Solution 5
# deactivate
# ```

# %% [markdown]
# ## Further Reading
# - Python Docs: venv — Creation of virtual environments
# - Python Packaging Authority: Managing Application Dependencies
