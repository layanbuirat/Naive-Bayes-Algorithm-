# Naive Bayes Algorithm

## 📌 Overview

Naive Bayes is a supervised machine learning algorithm based on applying Bayes' Theorem with strong (naive) independence assumptions between the features. It is primarily used for classification tasks such as spam filtering, sentiment analysis, and document categorization. Despite its simplicity, it often performs surprisingly well and is known for its speed and efficiency, especially with high-dimensional data.

---

## 📚 Table of Contents

1. [Core Concepts](#core-concepts)
   - [Bayes' Theorem](#bayes-theorem)
   - [Prior vs Posterior](#prior-vs-posterior)
   - [The "Naive" Assumption](#the-naive-assumption)
2. [Key Terminology](#key-terminology)
3. [Examples](#examples)
4. [Spam Classifier Project](#spam-classifier-project)
5. [Strengths and Weaknesses](#strengths-and-weaknesses)
6. [Quizzes & Solutions](#quizzes--solutions)
7. [Reflection Questions](#reflection-questions)
8. [Conclusion](#conclusion)

---

## 🧠 Core Concepts

### Bayes' Theorem

Bayes' Theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.

```
P(A|B) = P(B|A) * P(A) / P(B)
```

Where:
- **P(A|B)** is the posterior probability
- **P(A)** is the prior probability
- **P(B|A)** is the likelihood
- **P(B)** is the evidence

### Prior vs Posterior

| Term | Definition | Example |
|------|------------|---------|
| **Prior** | Initial guess before seeing new evidence | Probability person is Alex = 50% |
| **Posterior** | Updated probability after new evidence | Probability person is Alex given red sweater = 40% |

### The "Naive" Assumption

The "naive" aspect comes from the assumption that all features are **independent** of each other given the class.

**Example:** In spam detection, the algorithm assumes:
- P("money" AND "easy" | spam) = P("money" | spam) × P("easy" | spam)

This simplifies calculations drastically and works well in practice.

---

## 🔑 Key Terminology

| Term | Definition |
|------|------------|
| **Conditional Probability** | Probability of an event occurring given another event has occurred |
| **Naive Assumption** | The assumption that all features are independent of each other |
| **Posterior Probability** | Probability inferred *after* new information has arrived |
| **Prior Probability** | Probability inferred *before* new information has arrived |
| **Sensitivity** | Proportion of actual positives correctly identified |
| **Specificity** | Proportion of actual negatives correctly identified |

---

## 📊 Examples

### Example 1: Alex and Brenda

**Scenario:** Alex and Brenda work in an office. We see someone run by wearing a red sweater.

- **Prior:** Alex and Brenda equally likely (50% each)
- **New Info:** Alex wears red 2 days/week, Brenda wears red 3 days/week
- **Calculation:**
  - Total red days: 2 + 3 = 5 days
  - P(Alex | red) = 2/5 = 0.4 (40%)
  - P(Brenda | red) = 3/5 = 0.6 (60%)
- **Posterior:** 40% Alex, 60% Brenda

---

### Example 2: Medical Test

**Scenario:** A disease affects 1 in 10,000 people. A test is 99% accurate. If you test positive, what is the probability you are actually sick?

- **Prior:** P(Sick) = 0.0001
- **Likelihood:** P(Pos | Sick) = 0.99, P(Pos | Healthy) = 0.01
- **Calculation:**
  ```
  P(Sick | Pos) = (0.0001 × 0.99) / ((0.0001 × 0.99) + (0.9999 × 0.01)) ≈ 0.0098
  ```
- **Result:** Less than 1%! This is the **False Positive Paradox**.

---

### Example 3: Dice Problem

**Scenario:** A bag contains 3 standard dice (1-6) and 2 non-standard dice (faces: 2,3,3,4,4,5). A die is rolled and shows a 3. What is the probability it was standard?

- **Prior:** P(Standard) = 3/5, P(Non-standard) = 2/5
- **Likelihood:** P(3 | Standard) = 1/6, P(3 | Non-standard) = 2/6 = 1/3
- **Posterior:**
  ```
  P(Standard | 3) = (1/6 × 3/5) / ((1/6 × 3/5) + (1/3 × 2/5)) = (1/10) / (7/30) = 3/7 ≈ 0.429
  ```
- **Answer:** 0.429

---

## 📧 Spam Classifier Project

### Step 1: Data Preprocessing

```python
import pandas as pd

# Load dataset
df = pd.read_table('SMSSpamCollection', sep='\t', names=['label', 'sms_message'])

# Convert labels: ham=0, spam=1
df['label'] = df.label.map({'ham': 0, 'spam': 1})

print(df.shape)
df.head()
```

### Step 2: Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
count_vector = CountVectorizer()

# Fit and transform training data
training_data = count_vector.fit_transform(X_train)

# Transform testing data
testing_data = count_vector.transform(X_test)
```

### Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['sms_message'], 
    df['label'], 
    test_size=0.25, 
    random_state=42
)

print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')
```

### Step 4: Naive Bayes Implementation

```python
from sklearn.naive_bayes import MultinomialNB

# Initialize and train
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Predict
predictions = naive_bayes.predict(testing_data)
```

### Step 5: Model Evaluation

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

### Performance Summary

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

**Question:** Which statement about Prior and Posterior is correct?

- [ ] Prior refers to guesses after having complete information
- [ ] Posterior refers to guesses before new information arrives
- [x] Prior refers to guesses before complete information; Posterior refers to guesses after new information

**Answer:** The third option is correct.

---

### Quiz 2: Bayes Theorem

**Question:** Bayes theorem can be described as:

- [x] Based on known information, it can infer other information
- [ ] Based on inferred information, it can determine other information
- [ ] Based on known information, it can know other information

**Answer:** The first option is correct.

---

### Quiz 3: Probability Calculation

**Question:** What is the probability that an email contains the word 'easy', given that it is spam? (3 spam emails, 1 contains 'easy')

- [ ] 1/5
- [ ] 1/4
- [x] 1/3
- [ ] 2/3

**Answer:** 1/3

---

### Quiz 4: Probability Calculation

**Question:** What is the probability that an email contains the word 'money', given that it is spam? (3 spam emails, 2 contain 'money')

- [ ] 1/5
- [ ] 1/4
- [ ] 1/3
- [x] 2/3

**Answer:** 2/3

---

## 🤔 Reflection Questions

### Question 1

Which best describes a limitation of Naive Bayes?

- A. Cannot be used for binary classification
- B. Too complex to interpret
- C. Assumes feature independence, which may not hold in real data
- D. Requires deep neural networks to train

**Answer: C.** The "naive" assumption of feature independence is the main limitation.

---

### Question 2

When would Naive Bayes be an appropriate choice?

**Answer:** Naive Bayes is appropriate when:
- Speed is critical (trains very fast)
- Dataset has many features (text classification, NLP)
- Features are relatively independent
- Need a strong baseline model
- Problem is binary or multi-class classification

---

## 🏁 Conclusion

The Naive Bayes algorithm is a powerful, efficient, and easy-to-implement tool for classification. Its core strengths:

| Strength | Description |
|----------|-------------|
| ⚡ **Speed** | Extremely fast training and prediction |
| 📊 **High-dimensional data** | Handles thousands of features efficiently |
| 🛡️ **Robustness** | Performs well even with irrelevant features |
| 🎯 **Baseline model** | Excellent starting point for classification problems |

**Best for:**
- Spam detection
- Sentiment analysis
- Document categorization
- Medical diagnosis (with careful interpretation)

**Trade-offs:**
- ✅ Strengths: Speed, simplicity, robustness to irrelevant features
- ⚠️ Weaknesses: Assumes feature independence, struggles with correlated features

---

## 📖 Quick Reference

### Formula Summary

| Concept | Formula |
|---------|---------|
| **Bayes Theorem** | P(A\|B) = P(B\|A)P(A) / P(B) |
| **Naive Bayes Classifier** | ŷ = argmax P(y) ∏ P(xᵢ\|y) |
| **Laplace Smoothing** | P(xᵢ\|y) = (count(xᵢ,y) + α) / (count(y) + α × \|V\|) |

### Quick Start Code

```python
# Minimal implementation
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Prepare data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)
y = labels

# Train model
model = MultinomialNB()
model.fit(X, y)

# Predict
new_message = vectorizer.transform(["Win money now!"])
prediction = model.predict(new_message)
```


