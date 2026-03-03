# Zomato Sentiment Analysis using NLP & Machine Learning

## 📌 Project Overview

This project builds a binary sentiment classifier for Zomato restaurant reviews using classical NLP and Machine Learning techniques.

The model classifies reviews into:
- Positive
- Negative

Additionally, complaint themes are extracted from negative reviews.

---

## 🧠 Key Features

- Text preprocessing with NLTK
- Label noise reduction using VADER
- Word-level TF-IDF (1–3 grams)
- Character-level TF-IDF (3–5 grams)
- Additional numeric sentiment features
- Linear SVM with hyperparameter tuning
- Complaint theme detection

---

## 📊 Model Performance

- Accuracy: **65.8%**
- Macro F1-score: **65.3%**
- Balanced precision and recall

---

## 🏗️ Pipeline

1. Data Cleaning
2. Label Noise Removal
3. Feature Engineering
4. TF-IDF Vectorization
5. Hyperparameter Tuning (GridSearchCV)
6. Model Evaluation
7. Complaint Theme Analysis

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

---
