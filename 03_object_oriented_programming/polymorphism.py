# %% [markdown]
# # Python Polymorphism for Machine Learning
# 
# # Why this matters
# **Polymorphism** means "many forms." In Machine Learning code, polymorphism allows us to write functions or pipelines that can accept different types of objects (e.g., a Decision Tree, an SVM, or a Neural Net) and interact with them using a uniform interface (like calling `.predict()`). This makes ML pipelines flexible and extensible.
# 
# # Learning Objectives
# - Understand the concept of Polymorphism.
# - Learn how Python implements duck typing.
# - Understand how method overriding enables polymorphism.
# - See how polymorphism simplifies complex ML workflows like model ensembling or evaluation.
# 
# # Concept Explanation
# Polymorphism allows objects of different classes to be treated as objects of a common superclass, or simply guarantees that they share the same interface. In Python, this is often achieved through "duck typing" ("If it walks like a duck and quacks like a duck, it must be a duck") or through inheritance and overriding.
# 
# # Beginner Examples

# %%
# Example 1: Defining a common interface via Duck Typing
class SVMBasic:
    def predict(self, X):
        return [0, 1]

class DecisionTreeBasic:
    def predict(self, X):
        return [1, 0]

# %%
# Example 2: Polymorphic function
def evaluate_model(model, X):
    # This function works as long as the object has a 'predict' method
    predictions = model.predict(X)
    print(f"Predictions: {predictions}")

evaluate_model(SVMBasic(), [1, 2])
evaluate_model(DecisionTreeBasic(), [1, 2])

# %%
# Example 3: Polymorphism via Inheritance
class Transformer:
    def transform(self, X):
        pass

class MinMaxScaler(Transformer):
    def transform(self, X):
        return [x / 10.0 for x in X] # Mock logic

class StandardScaler(Transformer):
    def transform(self, X):
        return [x - 5.0 for x in X] # Mock logic

# %% [markdown]
# # Intermediate Examples

# %%
# Example 4: A pipeline of polymorphic transformers
def apply_pipeline(transformers, data):
    current_data = data
    for transformer in transformers:
        current_data = transformer.transform(current_data)
    return current_data

pipeline = [MinMaxScaler(), StandardScaler()]
print(apply_pipeline(pipeline, [10, 20]))

# %%
# Example 5: Polymorphism in Ensembles
class VotingClassifier:
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        # Calls predict on different model types uniformly
        return [model.predict(X) for model in self.models]

ensemble = VotingClassifier([SVMBasic(), DecisionTreeBasic()])
print(ensemble.predict([1,2]))

# %% [markdown]
# # Machine Learning Relevance
# Scikit-learn relies heavily on polymorphism. `cross_val_score`, `GridSearchCV`, and `Pipeline` can take *any* model object, as long as it exposes `fit`, `predict`, or `transform` methods. This means you can write a robust evaluation harness once, and plug in any new model architecture you invent without changing the evaluation code.
# 
# # Common Mistakes
# 1. Assuming objects must inherit from a common base class to be used polymorphically in Python (duck typing is often enough).
# 2. Inconsistent method signatures (e.g., one `.predict(X)` and another `.predict(X, batch_size)`), which breaks the polymorphic interface.
# 3. Over-engineering interfaces when a simple function would suffice.
# 4. Using `type(obj) == ...` checks instead of relying on the object's methods.
# 
# # Interview Questions
# 1. What is Polymorphism in the context of OOP?
# 2. What is "Duck Typing" in Python?
# 3. How does polymorphism reduce code coupling?
# 4. Provide an example of how polymorphism is used in standard ML libraries.
# 5. Why should you avoid using `isinstance()` checks when aiming for polymorphic design?
# 
# # Practice Problems
# 1. Create a `TextVectorizer` class and an `ImageVectorizer` class, both with an `encode(data)` method.
# 2. Write a function `process_data(vectorizer, data)` that calls `encode`.
# 3. Pass instances of both classes to `process_data` to demonstrate polymorphism.
# 4. Create a list of 3 different activation function objects (e.g., `ReLU`, `Sigmoid`), all having an `apply(x)` method.
# 5. Iterate through the list from problem 4 and call `apply(1.0)` on each.
# 
# # Solutions

# %%
# Solution 1
class TextVectorizer:
    def encode(self, data): return [len(data)]

class ImageVectorizer:
    def encode(self, data): return [sum(data)]

# %%
# Solution 2
def process_data(vectorizer, data):
    return vectorizer.encode(data)

# %%
# Solution 3
print(process_data(TextVectorizer(), "hello"))
print(process_data(ImageVectorizer(), [255, 128]))

# %%
# Solution 4
class ReLU:
    def apply(self, x): return max(0, x)

class Sigmoid:
    def apply(self, x): return 1 / (1 + 2.718**-x)

activations = [ReLU(), Sigmoid()]

# %%
# Solution 5
for act in activations:
    print(act.apply(1.0))

# %% [markdown]
# # Further Reading
# - Duck Typing glossary entry in Python Docs
# - Scikit-learn API design principles
