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
## 📖 Complete Code Reference

### Full Spam Classifier Implementation

```python
"""
Naive Bayes Spam Classifier - Complete Implementation
Author: Machine Learning Module
Date: 2026
"""

# ==================== 1. IMPORTS ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ==================== 2. LOAD AND EXPLORE DATA ====================
print("=" * 60)
print("NAIVE BAYES SPAM CLASSIFIER")
print("=" * 60)

# Load dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection', 
                   sep='\t', 
                   names=['label', 'sms_message'])

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📈 Class Distribution:")
print(df['label'].value_counts())

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Display sample
print("\n📝 Sample Data:")
print(df.head())

# ==================== 3. EXPLORATORY DATA ANALYSIS ====================
# Add message length feature
df['length'] = df['sms_message'].apply(len)

# Visualize message length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df[df['label'] == 0]['length'], bins=50, alpha=0.7, label='Ham', color='blue')
axes[0].hist(df[df['label'] == 1]['length'], bins=50, alpha=0.7, label='Spam', color='red')
axes[0].set_xlabel('Message Length')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Message Length Distribution by Class')
axes[0].legend()

# Box plot
df.boxplot(column='length', by='label', ax=axes[1])
axes[1].set_title('Message Length Box Plot')
axes[1].set_xlabel('Class (0=Ham, 1=Spam)')
axes[1].set_ylabel('Length')

plt.tight_layout()
plt.show()

print(f"\n📏 Average Message Length:")
print(f"Ham: {df[df['label'] == 0]['length'].mean():.1f} characters")
print(f"Spam: {df[df['label'] == 1]['length'].mean():.1f} characters")

# ==================== 4. TRAIN-TEST SPLIT ====================
X = df['sms_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n🔀 Data Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== 5. FEATURE EXTRACTION ====================
# Experiment with different vectorizers
vectorizers = {
    'CountVectorizer': CountVectorizer(stop_words='english', lowercase=True),
    'CountVectorizer (bigrams)': CountVectorizer(stop_words='english', ngram_range=(1, 2)),
    'TfidfVectorizer': TfidfVectorizer(stop_words='english', lowercase=True)
}

results = {}

for vec_name, vectorizer in vectorizers.items():
    print(f"\n🔄 Processing with {vec_name}...")
    
    # Transform data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   Feature dimensions: {X_train_vec.shape[1]}")
    
    # Train model
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = nb.predict(X_test_vec)
    
    # Evaluate
    results[vec_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# Display results
print("\n" + "=" * 60)
print("📊 VECTORIZER COMPARISON RESULTS")
print("=" * 60)

results_df = pd.DataFrame(results).T
print(results_df.round(3))

# ==================== 6. NAIVE BAYES VARIANT COMPARISON ====================
# Best vectorizer from previous step
best_vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = best_vectorizer.fit_transform(X_train)
X_test_vec = best_vectorizer.transform(X_test)

# Test different Naive Bayes variants
nb_variants = {
    'MultinomialNB': MultinomialNB(alpha=1.0),
    'BernoulliNB': BernoulliNB(alpha=1.0),
    'ComplementNB': ComplementNB(alpha=1.0)
}

variant_results = {}

for nb_name, nb_model in nb_variants.items():
    print(f"\n🔄 Training {nb_name}...")
    
    # Train and predict
    nb_model.fit(X_train_vec, y_train)
    y_pred = nb_model.predict(X_test_vec)
    
    # Store results
    variant_results[nb_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

print("\n" + "=" * 60)
print("📊 NAIVE BAYES VARIANT COMPARISON")
print("=" * 60)

variant_df = pd.DataFrame(variant_results).T
print(variant_df.round(3))

# ==================== 7. HYPERPARAMETER TUNING ====================
print("\n" + "=" * 60)
print("🔧 HYPERPARAMETER TUNING")
print("=" * 60)

# Tune alpha parameter
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_vec, y_train)

print(f"\n✅ Best alpha: {grid_search.best_params_['alpha']}")
print(f"✅ Best cross-validation F1 score: {grid_search.best_score_:.3f}")

# Train final model with best parameters
best_nb = MultinomialNB(alpha=grid_search.best_params_['alpha'])
best_nb.fit(X_train_vec, y_train)
y_pred_final = best_nb.predict(X_test_vec)

# ==================== 8. COMPREHENSIVE EVALUATION ====================
print("\n" + "=" * 60)
print("📊 FINAL MODEL EVALUATION")
print("=" * 60)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Ham', 'Spam']))

# Metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)

print(f"\n🎯 Final Metrics:")
print(f"   Accuracy:  {final_accuracy:.4f}")
print(f"   Precision: {final_precision:.4f}")
print(f"   Recall:    {final_recall:.4f}")
print(f"   F1 Score:  {final_f1:.4f}")

# ==================== 9. CONFUSION MATRIX ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('Confusion Matrix')

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[1])
axes[1].set_title('Normalized Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ==================== 10. ROC CURVE AND AUC ====================
# Get prediction probabilities
y_proba = best_nb.predict_proba(X_test_vec)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Naive Bayes (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Spam Classifier', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 11. PRECISION-RECALL CURVE ====================
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='Naive Bayes', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 12. CROSS-VALIDATION ====================
print("\n" + "=" * 60)
print("🔄 CROSS-VALIDATION RESULTS")
print("=" * 60)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(best_nb, X_train_vec, y_train, cv=5, scoring='f1')

print(f"\n📊 5-Fold Cross-Validation F1 Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score:.4f}")
print(f"\n   Mean F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==================== 13. TOP FEATURES ANALYSIS ====================
# Get feature names
feature_names = best_vectorizer.get_feature_names_out()

# Get feature log probabilities for spam class
spam_feature_probs = best_nb.feature_log_prob_[1, :]
ham_feature_probs = best_nb.feature_log_prob_[0, :]

# Get top features for spam
top_spam_idx = np.argsort(spam_feature_probs)[-20:][::-1]
top_spam_features = [(feature_names[i], np.exp(spam_feature_probs[i])) for i in top_spam_idx]

# Get top features for ham
top_ham_idx = np.argsort(ham_feature_probs)[-20:][::-1]
top_ham_features = [(feature_names[i], np.exp(ham_feature_probs[i])) for i in top_ham_idx]

print("\n" + "=" * 60)
print("🔍 TOP INDICATIVE FEATURES")
print("=" * 60)

print("\n🔥 Top 10 Spam Indicators:")
for word, prob in top_spam_features[:10]:
    print(f"   {word}: {prob:.4f}")

print("\n💚 Top 10 Ham Indicators:")
for word, prob in top_ham_features[:10]:
    print(f"   {word}: {prob:.4f}")

# ==================== 14. PREDICTION FUNCTION ====================
def predict_message(message, model, vectorizer):
    """
    Predict if a message is spam or ham.
    
    Parameters:
    -----------
    message : str
        The SMS message to classify
    model : trained classifier
        The trained Naive Bayes model
    vectorizer : CountVectorizer
        The fitted vectorizer
    
    Returns:
    --------
    tuple: (prediction, probability)
    """
    # Transform message
    message_vec = vectorizer.transform([message])
    
    # Get prediction and probability
    pred = model.predict(message_vec)[0]
    proba = model.predict_proba(message_vec)[0]
    
    # Result
    label = "SPAM" if pred == 1 else "HAM"
    confidence = proba[pred]
    
    return label, confidence

# Test predictions
print("\n" + "=" * 60)
print("📧 SAMPLE PREDICTIONS")
print("=" * 60)

test_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim now!",
    "Hey, are we still meeting for coffee tomorrow?",
    "URGENT: Your account has been compromised. Verify now!",
    "Thanks for the update, I'll see you there.",
    "FREE MONEY! Make $5000 per week working from home!!!",
    "Hi mom, can you call me when you get this?"
]

for msg in test_messages:
    label, confidence = predict_message(msg, best_nb, best_vectorizer)
    print(f"\n📨 Message: {msg}")
    print(f"   Prediction: {label} (confidence: {confidence:.2%})")

# ==================== 15. ERROR ANALYSIS ====================
print("\n" + "=" * 60)
print("⚠️ ERROR ANALYSIS")
print("=" * 60)

# Identify misclassified examples
misclassified_idx = np.where(y_pred_final != y_test)[0]

if len(misclassified_idx) > 0:
    print(f"\n🔴 Total misclassified: {len(misclassified_idx)} out of {len(y_test)} ({len(misclassified_idx)/len(y_test)*100:.1f}%)")
    
    # False positives (predicted spam, actually ham)
    fp_idx = np.where((y_pred_final == 1) & (y_test == 0))[0]
    print(f"\n❌ False Positives (Predicted Spam, Actually Ham): {len(fp_idx)}")
    for idx in fp_idx[:5]:  # Show first 5
        print(f"   - {X_test.iloc[idx][:100]}...")
    
    # False negatives (predicted ham, actually spam)
    fn_idx = np.where((y_pred_final == 0) & (y_test == 1))[0]
    print(f"\n❌ False Negatives (Predicted Ham, Actually Spam): {len(fn_idx)}")
    for idx in fn_idx[:5]:  # Show first 5
        print(f"   - {X_test.iloc[idx][:100]}...")
else:
    print("\n✅ No misclassified examples!")

# ==================== 16. MODEL SAVING FUNCTION ====================
import joblib
import os

def save_model(model, vectorizer, filename_prefix='naive_bayes_model'):
    """
    Save the trained model and vectorizer to disk.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model and vectorizer
    joblib.dump(model, f'models/{filename_prefix}_model.pkl')
    joblib.dump(vectorizer, f'models/{filename_prefix}_vectorizer.pkl')
    
    print(f"\n💾 Model saved to models/{filename_prefix}_model.pkl")
    print(f"💾 Vectorizer saved to models/{filename_prefix}_vectorizer.pkl")

def load_model(filename_prefix='naive_bayes_model'):
    """
    Load a trained model and vectorizer from disk.
    """
    model = joblib.load(f'models/{filename_prefix}_model.pkl')
    vectorizer = joblib.load(f'models/{filename_prefix}_vectorizer.pkl')
    
    print(f"\n📂 Model loaded from models/{filename_prefix}_model.pkl")
    print(f"📂 Vectorizer loaded from models/{filename_prefix}_vectorizer.pkl")
    
    return model, vectorizer

# Save the final model
save_model(best_nb, best_vectorizer)

# ==================== 17. PERFORMANCE SUMMARY ====================
print("\n" + "=" * 60)
print("📈 PERFORMANCE SUMMARY")
print("=" * 60)

summary_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Cross-Validation F1'],
    'Score': [final_accuracy, final_precision, final_recall, final_f1, auc, cv_scores.mean()],
    'Std Dev': ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', cv_scores.std()]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ==================== 18. CONCLUSION ====================
print("\n" + "=" * 60)
print("🎯 CONCLUSION")
print("=" * 60)

print("""
✅ SUCCESSFULLY IMPLEMENTED NAIVE BAYES SPAM CLASSIFIER

Key Achievements:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Achieved {:.2%} accuracy on test data
• F1 Score of {:.3f} indicates good balance of precision and recall
• AUC of {:.3f} shows excellent discrimination ability
• Training time of the model is extremely fast (< 0.1 seconds)
• Model is interpretable with clear feature importance

Performance Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Strong Performance: High precision means few false alarms
• Good Recall: Catches {:.1%} of actual spam messages
• Efficient: Suitable for real-time classification
• Interpretable: Can explain predictions based on keywords

Business Implications:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Reduces manual email filtering effort
• Protects users from phishing attempts
• Can be deployed in real-time email systems
• Requires minimal computational resources

Next Steps for Improvement:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Experiment with TF-IDF and n-grams
• Add more features (metadata, sender info)
• Implement ensemble methods
• Test on larger, more diverse datasets
• Deploy as web API
""".format(final_accuracy, final_f1, auc, final_recall * 100))

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
| **Log Probability** | \( \log P(y\|x) \propto \log P(y) + \sum \log P(x_i\|y) \) |

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

## 📚 Additional Learning Resources

### Books
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, Friedman
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"Introduction to Machine Learning with Python"** by Andreas Müller

### Online Courses
- **Stanford CS229** - Machine Learning
- **Fast.ai** - Practical Deep Learning
- **Coursera** - Machine Learning by Andrew Ng

### Research Papers
- Lewis, D. D. (1998). "Naive (Bayes) at forty: The independence assumption in information retrieval"
- Rish, I. (2001). "An empirical study of the naive Bayes classifier"
- Zhang, H. (2004). "The optimality of naive Bayes"

---

## 🤝 Contributing

We welcome contributions to improve this learning module! Areas for contribution:

- Additional examples and use cases
- Performance optimizations
- Multi-language support
- Interactive visualizations
- Bug fixes and documentation improvements

---

## 📄 License

This learning module is released under the MIT License. Feel free to use, modify, and distribute for educational and commercial purposes.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the SMS Spam Collection dataset
- **Scikit-learn community** for excellent machine learning tools
- **All contributors** who helped improve this learning material

---

## 📬 Contact

For questions, suggestions, or feedback:
- GitHub Issues: [https://github.com/layanbuirat/Naive-Bayes-Algorithm-]
- Email: mlayan774@gmail.com 

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-24 | Initial release |
| 1.1 | 2026-03-24 | Added advanced topics and code examples |
| 1.2 | 2026-03-24 | Completed comprehensive implementation |

---

## ⭐ Star this Repository

If you found this learning module helpful, please consider starring the repository and sharing it with others!

---

**"Simplicity is the ultimate sophistication."** - Leonardo da Vinci

*Naive Bayes proves that sometimes the simplest approach yields surprisingly powerful results.*

---

*Created with ❤️ for the machine learning community*
```

