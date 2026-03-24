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
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

Where:
- \( P(A|B) \) is the **posterior probability**
- \( P(A) \) is the **prior probability**
- \( P(B|A) \) is the **likelihood**
- \( P(B) \) is the **evidence**

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
| **Posterior Probability** | Probability inferred *after* new information has arrived \( P(A\|B) \) |
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

**Result:** Excellent performance with clear decision boundaries.

#### 2. Correlated/Noisy Features (⚠️ Weakness)

**Result:** Accuracy decreases when features are correlated, violating the independence assumption.

#### 3. High-Dimensional, Irrelevant Features (✅ Strength)

**Result:** Extremely fast training and robust to irrelevant features.

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

### Quiz 3: Probability of 'easy' in Spam

**Question:** What is the probability that an e-mail contains the word 'easy', given that it is spam?

**Data:** 3 spam emails, 1 contains 'easy'

- [ ] 1/5
- [ ] 1/4
- [x] 1/3
- [ ] 2/3

**Answer:** 1/3

---

### Quiz 4: Probability of 'money' in Spam

**Question:** What is the probability that an e-mail contains the word 'money', given that it is spam?

**Data:** 3 spam emails, 2 contain 'money'

- [ ] 1/5
- [ ] 1/4
- [ ] 1/3
- [x] 2/3

**Answer:** 2/3

---

### Quiz 5: Normalization

**Question:** What are the correct probabilities for spam and ham? In other words, what are two numbers that add to 1, and are in proportion to 1/12 and 1/40?

- [ ] 11/12 and 1/12
- [ ] 12/52 and 40/52
- [ ] 1/12 and 1/40
- [x] 10/13 and 3/13

**Answer:** 10/13 and 3/13

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

---

## 📖 Complete Code Reference

### Full Spam Classifier Implementation

```python
"""
Naive Bayes Spam Classifier - Complete Implementation
"""

# ==================== 1. IMPORTS ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# ==================== 2. LOAD AND EXPLORE DATA ====================
print("=" * 60)
print("NAIVE BAYES SPAM CLASSIFIER")
print("=" * 60)

# Load dataset
df = pd.read_table('SMSSpamCollection', 
                   sep='\t', 
                   names=['label', 'sms_message'])

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📈 Class Distribution:")
print(df['label'].value_counts())

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ==================== 3. TRAIN-TEST SPLIT ====================
X = df['sms_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n🔀 Data Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== 4. FEATURE EXTRACTION ====================
# Use CountVectorizer
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\n🔤 Feature dimensions: {X_train_vec.shape[1]}")

# ==================== 5. TRAIN NAIVE BAYES MODEL ====================
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_vec, y_train)

# ==================== 6. PREDICT AND EVALUATE ====================
y_pred = nb_model.predict(X_test_vec)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ==================== 7. CONFUSION MATRIX ====================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix - Spam Classifier\nAccuracy: {accuracy:.3f}')
plt.show()

# ==================== 8. CLASSIFICATION REPORT ====================
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ==================== 9. ROC CURVE ====================
y_proba = nb_model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Naive Bayes (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Spam Classifier')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ==================== 10. TOP FEATURES ====================
feature_names = vectorizer.get_feature_names_out()
spam_feature_probs = nb_model.feature_log_prob_[1, :]
top_spam_idx = np.argsort(spam_feature_probs)[-10:][::-1]

print("\n🔍 Top 10 Spam Indicators:")
for idx in top_spam_idx:
    print(f"   {feature_names[idx]}: {np.exp(spam_feature_probs[idx]):.4f}")

# ==================== 11. PREDICTION FUNCTION ====================
def predict_message(message):
    message_vec = vectorizer.transform([message])
    pred = nb_model.predict(message_vec)[0]
    proba = nb_model.predict_proba(message_vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    return label, proba[pred]

# Test predictions
print("\n" + "=" * 60)
print("📧 SAMPLE PREDICTIONS")
print("=" * 60)

test_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim now!",
    "Hey, are we still meeting for coffee tomorrow?",
    "URGENT: Your account has been compromised. Verify now!",
    "Thanks for the update, I'll see you there.",
]

for msg in test_messages:
    label, confidence = predict_message(msg)
    print(f"\n📨 Message: {msg}")
    print(f"   Prediction: {label} (confidence: {confidence:.2%})")

print("\n" + "=" * 60)
print("🏁 END OF IMPLEMENTATION")
print("=" * 60)
```

---

## 📝 Quick Reference Card

### Formula Summary

| Concept | Formula |
|---------|---------|
| **Bayes Theorem** | \( P(A\|B) = \frac{P(B\|A)P(A)}{P(B)} \) |
| **Naive Bayes Classifier** | \( \hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i \| y) \) |
| **Laplace Smoothing** | \( P(x_i\|y) = \frac{\text{count}(x_i,y) + \alpha}{\text{count}(y) + \alpha \times \|V\|} \) |

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

---

**"Simplicity is the ultimate sophistication."** - Leonardo da Vinci

*Naive Bayes proves that sometimes the simplest approach yields surprisingly powerful results.*

---

*Created with ❤️ for the machine learning community*
```


