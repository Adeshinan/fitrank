# Classification or regression?

FitRank asks you to declare `task:` in the YAML. This guide shows how to
recognise the task of a problem, and which other kinds of tasks exist.

## The basic rule

Look at **the nature of what you predict**, not its format.

| You predict… | Task | Examples |
|---|---|---|
| **A category** (one label from a fixed list) | **Classification** | Survived or not? Spam or not? Which species of flower? |
| **A quantity** (a number on a continuous scale) | **Regression** | Which price? Which temperature tomorrow? How many minutes of travel? |

## Three questions to decide

1. **Does an average make sense?**
    - The average of €200,000 and €300,000 is €250,000, which makes sense → **regression**
    - The average of "cat" and "dog" makes no sense → **classification**

2. **Does the size of an error matter?**
    - Predicting €210,000 instead of €200,000 is nearly right, €900,000 is very wrong → **regression**
    - Predicting "dog" instead of "cat" is simply wrong, there is no "nearly" → **classification**

3. **Can the model output a value never seen before?**
    - A price of €237,451 may appear nowhere in the data, yet it is a valid answer → **regression**
    - An invented species of flower is not a valid answer → **classification**

!!! warning "Beware of the format"
    A number is **not always** a quantity. A postcode (75001), a ticket class
    (1, 2, 3) or a 0/1 label are **categories** written with digits.

    Hint: few possible values, and no calculation makes sense between them
    → it is classification.

## Borderline cases

| Target | Rather… | Why |
|---|---|---|
| A rating from 1 to 5 stars | It depends | There is an order, but not a true continuous scale. This is **ordinal regression** (see below). |
| Number of children (0, 1, 2…) | Regression | It is a count: the average makes sense. A dedicated variant exists, **Poisson regression**. |
| Age in brackets ("18-25", "26-35") | Classification | They are categories, even if they come from a number. |
| Probability of a payment default | Classification | The classes are "default / no default", and the model gives a probability for each class. |

## Variants of classification and regression

- **Binary classification**: two classes (yes / no).
- **Multiclass classification**: more than two classes, one answer per example (iris: setosa, versicolor or virginica).
- **Multilabel classification**: several labels at once (a film that is both "comedy" and "romance").
- **Multi-output regression**: several numbers at once (latitude and longitude).

FitRank handles binary and multiclass classification, and single-output
regression.

## Other kinds of tasks

Classification and regression belong to **supervised learning**: the examples
come with the right answer (the *target*). Other families exist.

### Supervised, but different

| Task | Example |
|---|---|
| Ordinal regression | Predict a rating from 1 to 5 while respecting the order |
| Time series forecasting | Next month's sales from the past months |
| Ranking | Order search results |
| Survival analysis | How long before a customer leaves or a machine breaks down |

### Unsupervised (no target, looking for a structure)

| Task | Example |
|---|---|
| Clustering | Group customers with similar behaviour |
| Dimensionality reduction | Summarise 50 columns in 3 (e.g. PCA) |
| Anomaly detection | Spot a fraudulent bank transaction |

### Other families

- **Reinforcement learning**: an agent learns by trial and error, with rewards (games, robots).
- **Generative models**: produce new content (text, images).
- **Semi-supervised learning**: a few examples have a target, most do not.

## The reflex to keep

> Before any model, ask: **"is my answer a label or a quantity?"**
>
> - A label → **classification**
> - A quantity → **regression**

On tabular data, this question is enough in the vast majority of cases.

### Why FitRank does not guess

The task cannot be deduced from the target alone. A rating from 1 to 5 can be
a classification **or** a regression: it is a choice about the problem, not
about the data. That is why FitRank asks you to declare `task:` explicitly. It
only warns, without stopping, when the target looks unlikely for the task
(see the [configuration reference](../configuration.md#task)).
