| Name | NRP | 
| ---- | --- | 
| William Hans Chandra  | 5025241138 |
| Frenaldy Bestabba Hasugian  | 5025241156 |
| Muh. Aqil Alqadri Syahid  | 5025241160 |
| Kadek Andra Wikanjaya Putra  | 5025241187 |


# Naive Bayes Implementation

## What is Naive Bayes?
Naive Bayes is a probability-based classification algorithm that uses the **Bayes Theorem** to determine the likelihood that a given data point belongs to a particular class. This algorithm is widely used in various classification tasks such as spam detection, sentiment analysis, and text classification.

The algorithm is called *naive* because it assumes that every feature in the data is independent of one another, even though in practice these features are not always truly independent.

---

## Bayes Theorem
**Bayes Theorem** is a probability principle used to calculate the likelihood of an event based on new information. In other words, it helps estimate how probable an event is given certain conditions or data.

The foundation of the Naive Bayes algorithm is Bayes Theorem, formulated as follows:

$$
P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}
$$

Where:

- $P(C|X)$ : probability that data $X$ belongs to class $C$ (posterior probability)  
- $P(X|C)$ : probability that feature $X$ appears given class $C$ (likelihood)  
- $P(C)$ : initial probability of class $C$ (prior probability)  
- $P(X)$ : probability of data $X$ occurring

---

## Independence Assumption in Naive Bayes
Naive Bayes assumes that every feature is independent. Under this assumption, the probability of multiple features can be calculated as the product of the individual feature probabilities.

$$
P(X_1, X_2, ..., X_n | C) =
P(X_1|C) \times P(X_2|C) \times ... \times P(X_n|C)
$$

Therefore, the Naive Bayes classification formula becomes:

$$
P(C|X) \propto P(C) \prod_{i=1}^{n} P(X_i|C)
$$

---

## Steps of the Naive Bayes Algorithm

### 1. Calculate Prior Probability
Calculate the initial probability of each class based on the number of data points in the dataset.

For example:

$$
P(C) = \frac{\text{number of data points in class } C}{\text{total data points}}
$$

---

### 2. Calculate Likelihood
Calculate the probability of each feature given a particular class.

$$
P(X_i | C)
$$

In text classification, features are typically words.

---

### 3. Calculate Posterior Probability
For new data, the probability of each class is calculated by multiplying the prior probability by the probabilities of the features appearing in that class.

$$
P(C|X) = P(C) \times P(X_1|C) \times P(X_2|C) \times ... \times P(X_n|C)
$$

---

### 4. Determine the Class
The class with the highest probability is selected as the classification result.

$$
\hat{C} = \arg\max P(C|X)
$$

---

## Simple Example
Suppose we have the following comment: "you are stupid"

The algorithm will calculate:

$$
P(Toxic) \times P(you|Toxic) \times P(are|Toxic) \times P(stupid|Toxic)
$$

$$
P(Normal) \times P(you|Normal) \times P(are|Normal) \times P(stupid|Normal)
$$

It then selects the class with the highest probability as the classification result.

---
## Dataset
The dataset contains text comments labeled with three types of toxic content: threat, insult, and sexual harassment. Each comment is categorized according to the type of toxicity it represents.

To classify these comments, the **Multinomial Naive Bayes** algorithm is used. This method works by calculating the likelihood of each word appearing in each class, combined with the prior probability of each class. For a given comment, it computes the posterior probability for all three classes and predicts the class with the highest probability.

This approach is particularly effective for text data because it captures the distribution of words across classes, allowing the model to distinguish between different forms of toxicity. Preprocessing steps such as lowercasing, tokenization, and stopword removal are applied to clean the text and improve model performance.

## Workflow Overview

The classification process follows these systematic steps:

### 1. Preprocessing & Normalization
To ensure the model focuses on relevant data, we perform:
* **Lowercase Conversion**: Standardizing all characters.
* **Symbol Mapping**: Converting symbols/numbers that mimic letters (e.g., `@` to `a`, `1` to `i`, `$` to `s`).
* **Punctuation Removal**: Removing noise while maintaining the core text.
* **Stopwords Removal**: Filtering out common words (e.g., "is", "the") using the NLTK library.

### 2. Feature Engineering (Emoji Detection)
Emojis are treated as distinct features. We categorize specific emojis into three classes:
* **Sexual Harassment**: 🍆, 🍑, 👅, 💦, etc.
* **Insult**: 💩, 🖕, 🤡, 🐷, etc.
* **Threat**: 🔫, 🔪, 💀, 🩸, etc.

### 3. Classification Algorithm
This project implements the **Naive Bayes** algorithm based on **Bayes' Theorem**:
The algorithm calculates the posterior probability for each category and assigns the class with the highest probability.

### 4. Visualization
We use **Word Clouds** to visually analyze the most dominant terms used in each toxic category.

# 5. Modelling & Evaluation

## Modelling Phase
The modelling phase in this project utilizes the **Naive Bayes algorithm**, a probabilistic classifier based on Bayes' Theorem. This algorithm is chosen for its high efficiency and effectiveness in **Natural Language Processing (NLP)** tasks.

### Algorithm Logic
The model calculates the probability of a comment belonging to a specific category (**Insult, Threat, or Sexual Harassment**) based on the words and emojis it contains.

### Feature Independence
The "Naive" assumption treats every feature (word or emoji) as independent of others. This speeds up computation while maintaining competitive accuracy for text-based datasets.

### Hybrid Feature Set
Unlike standard text classifiers, this model incorporates a **custom feature set**, combining:
- **Textual tokens** (processed using TF-IDF)  
- **Numerical emoji counts**  

This allows the model to capture toxic intent that is often hidden in visual symbols, not just plain text.

### Training Process
- Dataset is split into **training** and **testing** sets.  
- During training, the model learns:
  - **Likelihood:** Probability of each word/emoji appearing in each toxic class  
  - **Prior Probability:** Base probability of each class based on dataset distribution  

---

## Evaluation Phase
To ensure reliable classifications, performance is evaluated using **quantitative** and **qualitative** metrics:

### 1. Accuracy Score
- Measures the overall percentage of correct predictions out of total samples  
- Provides a quick overview of the model's general performance
  a

### 2. Confusion Matrix
- Detailed table describing classification performance  
- Shows where the model is "confused" (e.g., misclassifying a Threat as an Insult)  
- Tracks:
  - **True Positives (TP):** Correctly identified toxic comments  
  - **False Positives (FP):** Normal comments incorrectly flagged as toxic  
  - **False Negatives (FN):** Toxic comments the model failed to detect  

### 3. Classification Report

- **Precision:** Accuracy of positive predictions (minimizes false alarms)  
- **Recall:** Ability to find all toxic comments (minimizes missed detections)  
- **F1-Score:** Harmonic mean of Precision and Recall, balancing both metrics; crucial for imbalanced datasets  

### 4. Qualitative Analysis (Word Clouds)
![image alt](https://github.com/kebojuara/probstat-tugas-2/blob/6417be2a94f7a59790ac51ae919f2346c3010e2a/WhatsApp%20Image%202026-03-14%20at%2019.22.36.jpeg)
- Generates **Word Clouds** for each category as a visual sanity check  
- Example: If the "Sexual Harassment" cloud is dominated by expected emojis and keywords, it validates that the model is learning correct semantic patterns
