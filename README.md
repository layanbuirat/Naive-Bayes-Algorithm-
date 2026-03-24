# README.md

```markdown
# Naive Bayes Algorithm

## 📌 Overview

Naive Bayes is a supervised machine learning algorithm based on applying Bayes' Theorem with strong (naive) independence assumptions between the features. It is primarily used for classification tasks such as spam filtering, sentiment analysis, and document categorization. Despite its simplicity, it often performs surprisingly well and is known for its speed and efficiency, especially with high-dimensional data.

---

## 📚 Table of Contents

1. [Core Concepts](#core-concepts)
   - [Bayes Theorem](#bayes-theorem)
   - [Prior vs Posterior](#prior-vs-posterior)
   - [The "Naive" Assumption](#the-naive-assumption)
2. [Key Terminology](#key-terminology)
3. [Examples](#examples)
   - [Example 1: Alex and Brenda (Intuition)](#example-1-alex-and-brenda-intuition)
   - [Example 2: Medical Test (False Positives)](#example-2-medical-test-false-positives)
   - [Example 3: Dice Problem](#example-3-dice-problem)
4. [Spam Classifier Project](#spam-classifier-project)
   - [Step 1: Data Preprocessing](#step-1-data-preprocessing)
   - [Step 2: Bag of Words (BoW)](#step-2-bag-of-words-bow)
   - [Step 3: Train-Test Split](#step-3-train-test-split)
   - [Step 4: Naive Bayes Implementation](#step-4-naive-bayes-implementation)
   - [Step 5: Model Evaluation](#step-5-model-evaluation)
5. [Strengths and Weaknesses](#strengths-and-weaknesses)
   - [Performance Experiments](#performance-experiments)
6. [Quizzes & Solutions](#quizzes--solutions)
7. [Reflection Questions](#reflection-questions)
8. [Conclusion](#conclusion)

---

## 🧠 Core Concepts

### Bayes Theorem

Bayes' Theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.

\[
P(A|R) = \frac{P(R|A) \cdot P(A)}{P(R)}
\]

Where:
- \( P(A|R) \) is the **posterior probability**
- \( P(A) \) is the **prior probability**
- \( P(R|A) \) is the **likelihood**
- \( P(R) \) is the **evidence**

### Prior vs Posterior

| Term | Definition | Example |
|------|------------|---------|
| **Prior** | Initial guess before seeing new evidence | Probability person is Alex = 50% |
| **Posterior** | Updated probability after new evidence | Probability person is Alex given red sweater = 40% |

### The "Naive" Assumption

The "naive" aspect comes from the assumption that all features are **independent** of each other given the class.

**Example:** In spam detection, the algorithm assumes:
- \( P(\text{"money"} \cap \text{"easy"} \mid \text{spam}) = P(\text{"money"} \mid \text{spam}) \times P(\text{"easy"} \mid \text{spam}) \)

This simplifies calculations drastically and works well in practice, even though words may actually be correlated.

---

## 🔑 Key Terminology

| Term | Definition |
|------|------------|
| **Conditional Probability** | Probability of an event occurring given another event has occurred |
| **Naive Assumption** | The assumption that all features are independent of each other |
| **Posterior Probability** | Probability inferred *after* new information has arrived \( P(A\|R) \) |
| **Prior Probability** | Probability inferred *before* new information has arrived \( P(A) \) |
| **Sensitivity (True Positive Rate)** | Proportion of actual positives correctly identified |
| **Specificity (True Negative Rate)** | Proportion of actual negatives correctly identified |

---

## 📊 Examples

### Example 1: Alex and Brenda (Intuition)

**Scenario:** Alex and Brenda work in an office. We see someone run by quickly wearing a red sweater.

- **Prior:** Alex and Brenda are in the office equally (50% each)
- **New Info:** Alex wears red 2 days/week. Brenda wears red 3 days/week
- **Calculation:**
  - Total "red days": 2 (Alex) + 3 (Brenda) = 5 days
  - \( P(\text{Alex} \mid \text{red}) = \frac{2}{5} = 0.4 \)
  - \( P(\text{Brenda} \mid \text{red}) = \frac{3}{5} = 0.6 \)
- **Posterior:** 40% Alex, 60% Brenda

---

### Example 2: Medical Test (False Positives)

**Scenario:** A disease affects 1 in 10,000 people. A test is 99% accurate (sensitivity = specificity = 0.99). If you test positive, what is the probability you are actually sick?

- **Prior:** \( P(\text{Sick}) = 0.0001 \)
- **Likelihood:** \( P(\text{Pos} \mid \text{Sick}) = 0.99 \), \( P(\text{Pos} \mid \text{Healthy}) = 0.01 \)
- **Calculation:**
  \[
  P(\text{Sick} \mid \text{Pos}) = \frac{0.0001 \times 0.99}{(0.0001 \times 0.99) + (0.9999 \times 0.01)} \approx 0.0098
  \]
- **Result:** Less than 1%! This illustrates the **False Positive Paradox** — the large number of healthy people causes many false positives.

---

### Example 3: Dice Problem

**Scenario:** A bag contains:
- 3 standard dice (faces: 1-6)
- 2 non-standard dice (faces: [2, 3, 3, 4, 4, 5])

A die is drawn and rolled, showing a 3. What is the probability it was a standard die?

- **Prior:** \( P(S) = \frac{3}{5} \), \( P(N) = \frac{2}{5} \)
- **Likelihood:** \( P(3 \mid S) = \frac{1}{6} \), \( P(3 \mid N) = \frac{2}{6} = \frac{1}{3} \)
- **Posterior:**
  \[
  P(S \mid 3) = \frac{\frac{1}{6} \times \frac{3}{5}}{\left(\frac{1}{6} \times \frac{3}{5}\right) + \left(\frac{1}{3} \times \frac{2}{5}\right)} = \frac{\frac{1}{10}}{\frac{7}{30}} = \frac{3}{7} \approx 0.429
  \]
- **Answer:** 0.429 (rounded to 3 decimal places)

---

## 📧 Spam Classifier Project

This project walks through building a Spam/Ham classifier using Naive Bayes and the Bag of Words model.

### Step 1: Data Preprocessing

Load the dataset (`SMSSpamCollection`) and convert labels to binary values.

```python
import pandas as pd

# Load dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection', 
                   sep='\t', 
                   names=['label', 'sms_message'])

# Convert labels: ham=0, spam=1
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Check the data
print(df.shape)
df.head()
```

### Step 2: Bag of Words (BoW)

Convert text messages into numerical feature vectors by counting word frequencies.

#### From Scratch Implementation:

```python
# Convert to lowercase
documents = ['Hello, how are you!', 'Win money, win from home.', 
             'Call me now.', 'Hello, Call hello you tomorrow?']

lower_case_documents = [doc.lower() for doc in documents]

# Remove punctuation
import string
sans_punctuation_documents = []
for doc in lower_case_documents:
    sans_punctuation_documents.append(doc.translate(str.maketrans('', '', string.punctuation)))

# Tokenize
preprocessed_documents = [doc.split() for doc in sans_punctuation_documents]

# Count frequencies
from collections import Counter
frequency_list = [Counter(doc) for doc in preprocessed_documents]
```

#### Using Scikit-Learn:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
count_vector = CountVectorizer()

# Fit and transform training data
training_data = count_vector.fit_transform(X_train)

# Transform testing data (only transform, no fit)
testing_data = count_vector.transform(X_test)
```

### Step 3: Train-Test Split

Split data to evaluate model performance on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['sms_message'], 
    df['label'], 
    test_size=0.25, 
    random_state=42
)

print(f'Training set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')
```

### Step 4: Naive Bayes Implementation

Train a Multinomial Naive Bayes model.

```python
from sklearn.naive_bayes import MultinomialNB

# Initialize the classifier
naive_bayes = MultinomialNB()

# Train the model
naive_bayes.fit(training_data, y_train)

# Make predictions
predictions = naive_bayes.predict(testing_data)
```

### Step 5: Model Evaluation

Evaluate using accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(naive_bayes, testing_data, y_test)
plt.title(f'Confusion Matrix - Spam Classifier\nAccuracy: {accuracy:.3f}')
plt.show()
```

**Sample Output:**
```
Accuracy: 0.987
Precision: 0.972
Recall: 0.940
F1 Score: 0.956
```

---

## ⚡ Strengths and Weaknesses

### Performance Experiments

#### 1. Clean, Independent Features (✅ Strength)

**Data:** Two informative, independent features with clear separation.

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

# Generate clean separable data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, class_sep=2.0, random_state=0)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Results: High accuracy, clear decision boundary
```

**Result:** Excellent performance with clear decision boundaries.

---

#### 2. Correlated/Noisy Features (⚠️ Weakness)

**Data:** Features are redundant or correlated; class boundaries overlap.

```python
# Generate data with correlated features
X_corr, y_corr = make_classification(n_samples=200, n_features=2, 
                                     n_informative=1, n_redundant=1,
                                     class_sep=0.8, random_state=1)

# Train model
model_corr = GaussianNB()
model_corr.fit(X_train_corr, y_train_corr)

# Results: Lower accuracy, ambiguous boundaries
```

**Result:** Accuracy decreases when features are correlated, violating the independence assumption.

---

#### 3. High-Dimensional, Irrelevant Features (✅ Strength)

**Data:** 100 features, but only 2 are informative.

```python
import time

# Generate high-dimensional data
X_highdim, y_highdim = make_classification(
    n_samples=1000, n_features=100, 
    n_informative=2, n_redundant=10,
    n_clusters_per_class=1, random_state=42
)

# Measure training time
start_time = time.time()
model_hd = GaussianNB()
model_hd.fit(X_train_hd, y_train_hd)
train_time = time.time() - start_time

print(f'Training time: {train_time:.4f} seconds')
print(f'Accuracy: {accuracy_score(y_test_hd, y_pred_hd):.3f}')
```

**Sample Output:**
```
Training time: 0.0022 seconds
Accuracy: 0.86
```

**Result:** Extremely fast training and robust to irrelevant features.

---

### Summary Table

| Aspect | Performance |
|--------|-------------|
| **Speed** | ✅ Excellent — trains in milliseconds |
| **High-dimensional data** | ✅ Very good — handles thousands of features |
| **Irrelevant features** | ✅ Robust — unaffected by noise |
| **Independent features** | ✅ Outstanding — optimal performance |
| **Correlated features** | ⚠️ Weak — assumption violation reduces accuracy |
| **Complex boundaries** | ⚠️ Weak — linear decision boundaries |

---

## 📝 Quizzes & Solutions

### Quiz 1: Prior & Posterior

**Question:** Which of the following is an accurate statement related to Naive Bayes?

- [ ] Prior refers to guesses we make after having complete information
- [ ] Posterior refers to guesses we've inferred before new information has arrived
- [x] Prior refers to guesses we make before having complete information; Posterior refers to guesses we've inferred after new information has arrived

**Answer:** The third option is correct.

---

### Quiz 2: Bayes Theorem

**Question:** Bayes theorem can be described as follows:

- [x] Based on information that is known, it can infer other information
- [ ] Based on information that is inferred, it can determine other information
- [ ] Based on information that is known, it can know other information

**Answer:** The first option is correct.

---

### Quiz 3: Discarding Scenarios

**Question:** During the explanation above, once 4 possible scenarios are determined, 2 of them are discarded. Why is this appropriate?

- [ ] Out of the 4 scenarios, only 2 of them included both Alex and Brenda, so those scenarios are irrelevant
- [x] Out of the 4 scenarios, only 2 of them included someone wearing a red sweater, so those scenarios are irrelevant
- [ ] Out of the 4 scenarios, only 1 of them included probabilities > 50% for Alex or Brenda, so those scenarios are irrelevant

**Answer:** The second option is correct. We discard scenarios where the person is not wearing red.

---

### Quiz 4: Probability of 'easy' in Spam

**Question:** What is the probability that an e-mail contains the word 'easy', given that it is spam?

**Data:** 3 spam emails, 1 contains 'easy'

- [ ] 1/5
- [ ] 1/4
- [x] 1/3
- [ ] 2/3
- [ ] 4/5

**Answer:** 1/3

---

### Quiz 5: Probability of 'money' in Spam

**Question:** What is the probability that an e-mail contains the word 'money', given that it is spam?

**Data:** 3 spam emails, 2 contain 'money'

- [ ] 1/5
- [ ] 1/4
- [ ] 1/3
- [x] 2/3
- [ ] 4/5

**Answer:** 2/3

---

### Quiz 6: Normalization

**Question:** What are the correct probabilities for spam and ham? In other words, what are two numbers that add to 1, and are in proportion to 1/12 and 1/40?

- [ ] 11/12 and 1/12
- [ ] 12/52 and 40/52
- [ ] 1/12 and 1/40
- [x] 10/13 and 3/13

**Answer:** 10/13 and 3/13

**Explanation:**
\[
\frac{1/12}{1/12 + 1/40} = \frac{1/12}{10/120 + 3/120} = \frac{1/12}{13/120} = \frac{10}{13}
\]
\[
\frac{1/40}{1/12 + 1/40} = \frac{1/40}{13/120} = \frac{3}{13}
\]

---

## 🤔 Reflection Questions

### Question 1

Which of the following best describes a limitation of Naive Bayes?

- A. It cannot be used for binary classification
- B. It is too complex to interpret
- C. It assumes feature independence, which may not hold in real data
- D. It requires deep neural networks to train

**Answer: C.** The "naive" assumption of feature independence is the main limitation of the algorithm.

---

### Question 2

When would Naive Bayes be an appropriate choice for a classification problem?

**Answer:** Naive Bayes is appropriate when:
- Speed is critical (trains very fast)
- The dataset has many features (text classification, NLP)
- Features are relatively independent
- You need a strong baseline model
- The problem is binary or multi-class classification with discrete features

---

## 🏁 Conclusion

The Naive Bayes algorithm is a powerful, efficient, and easy-to-implement tool for classification. Its core strengths lie in:

| Strength | Description |
|----------|-------------|
| ⚡ **Speed** | Extremely fast training and prediction |
| 📊 **High-dimensional data** | Handles thousands of features efficiently |
| 🛡️ **Robustness** | Performs well even with irrelevant features |
| 🎯 **Baseline model** | Excellent starting point for classification problems |

While the independence assumption is often violated in practice, the algorithm remains surprisingly effective, particularly for tasks like:
- **Spam detection**
- **Sentiment analysis**
- **Document categorization**
- **Medical diagnosis** (with careful interpretation)

The trade-offs are clear:
- ✅ **Strengths:** Speed, simplicity, robustness to irrelevant features
- ⚠️ **Weaknesses:** Assumes feature independence, struggles with correlated features

Overall, Naive Bayes is a gem of an algorithm that balances simplicity with effectiveness, making it a valuable addition to any data scientist's toolkit.

---

## 📖 Additional Resources

- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Bayes' Theorem - Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/bayes-theorem/)

---

```
