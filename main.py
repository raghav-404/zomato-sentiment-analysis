import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, ConfusionMatrixDisplay
from scipy.sparse import hstack, csr_matrix


# 1. Data Loading

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

df = pd.read_csv("zomato_reviews.csv")
df = df.dropna(subset=["review"])

def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["rating"].apply(label_sentiment)

# Keep only binary labels
df = df[df["sentiment"] != "Neutral"].copy()


# 2. Remove Label Noise using VADER

sia = SentimentIntensityAnalyzer()

df["vader_score"] = df["review"].apply(
    lambda x: sia.polarity_scores(str(x))["compound"]
)

noise_mask = (
    ((df["sentiment"] == "Negative") & (df["vader_score"] > 0.5)) |
    ((df["sentiment"] == "Positive") & (df["vader_score"] < -0.5))
)

df = df[~noise_mask].copy()


# 3. Text Cleaning

NEGATIONS = {
    "not","no","never","nor","neither","nothing",
    "cannot","without","hardly","barely","scarcely"
}

stop_words = set(stopwords.words("english")) - NEGATIONS
lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "i'm": "i am",
    "it's": "it is"
}

def clean_text(text):
    text = str(text).lower()

    for c, r in CONTRACTIONS.items():
        text = text.replace(c, r)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if (word not in stop_words or word in NEGATIONS) and len(word) > 1
    ]

    return " ".join(tokens)

df["cleaned"] = df["review"].apply(clean_text)


# 4. Feature Engineering

def extract_numeric_features(data):
    features = pd.DataFrame()

    vader_scores = data["review"].apply(
        lambda x: sia.polarity_scores(str(x))
    )

    features["vader_compound"] = vader_scores.apply(lambda x: x["compound"])
    features["vader_pos"] = vader_scores.apply(lambda x: x["pos"])
    features["vader_neg"] = vader_scores.apply(lambda x: x["neg"])
    features["vader_neu"] = vader_scores.apply(lambda x: x["neu"])

    features["review_len"] = data["review"].str.split().str.len()
    features["char_len"] = data["review"].str.len()
    features["exclamation_count"] = data["review"].str.count("!")
    features["question_count"] = data["review"].str.count(r"\?")
    features["caps_ratio"] = data["review"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )

    features["negation_density"] = data["cleaned"].apply(
        lambda x: sum(1 for w in x.split() if w in NEGATIONS) /
                  max(len(x.split()), 1)
    )

    return features.fillna(0)

numeric_features = extract_numeric_features(df)


# 5. Train-Test Split

X_text = df["cleaned"]
X_num = numeric_features
y = df["sentiment"]

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_num, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. TF-IDF Vectorization

word_vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    min_df=2,
    sublinear_tf=True
)

char_vectorizer = TfidfVectorizer(
    max_features=4000,
    ngram_range=(3, 5),
    min_df=3,
    analyzer="char_wb",
    sublinear_tf=True
)

X_word_train = word_vectorizer.fit_transform(X_text_train)
X_word_test = word_vectorizer.transform(X_text_test)

X_char_train = char_vectorizer.fit_transform(X_text_train)
X_char_test = char_vectorizer.transform(X_text_test)

# Scale numeric features
scaler = StandardScaler()
X_num_train_scaled = csr_matrix(scaler.fit_transform(X_num_train))
X_num_test_scaled = csr_matrix(scaler.transform(X_num_test))

# Combine all features
X_train = hstack([X_word_train, X_char_train, X_num_train_scaled])
X_test = hstack([X_word_test, X_char_test, X_num_test_scaled])


# 7. Model Training (Hyperparameter Tuning)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}

grid = GridSearchCV(
    LinearSVC(class_weight="balanced", max_iter=3000),
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_C = grid.best_params_["C"]


# 8. Evaluation

y_pred = best_model.predict(X_test)

print("Best C:", best_C)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-macro:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))


# 9. Complaint Theme Analysis (Negative Reviews)

themes = {
    "Delivery Issues": ["late","delayed","rider","waiting","slow"],
    "Food Quality": ["cold","bad","smell","stale","raw","burnt"],
    "Pricing": ["expensive","price","overpriced","cost"],
    "Customer Service": ["rude","support","staff","cancel"],
    "Wrong Order": ["wrong","missing","mistake"]
}

def detect_theme(text):
    for category, keywords in themes.items():
        if any(word in text.lower() for word in keywords):
            return category
    return "Other"

neg_reviews = df[df["sentiment"] == "Negative"].copy()
neg_reviews["theme"] = neg_reviews["cleaned"].apply(detect_theme)


# 10. Visualization

plt.figure(figsize=(8,6))
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test, cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(8,6))
neg_reviews["theme"].value_counts().plot(kind="barh")
plt.title("Complaint Categories")
plt.xlabel("Count")
plt.show()
