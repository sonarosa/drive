!pip install numpy pandas scikit-learn nltk
import nltk
nltk.download('movie_reviews')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import movie_reviews

positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')

documents = [(movie_reviews.raw(fileid), 'pos') for fileid in positive_fileids]
documents.extend([(movie_reviews.raw(fileid), 'neg') for fileid in negative_fileids])

df = pd.DataFrame(documents, columns=['text', 'label'])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

class NaiveBayes:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_counts = {c: 0 for c in self.classes}
        self.feature_counts = {c: np.zeros(X.shape[1]) for c in self.classes}

        for i, c in enumerate(y):
            self.class_counts[c] += 1
            self.feature_counts[c] += X[i].toarray()[0]

        self.total_documents = len(y)
        self.vocab_size = X.shape[1]

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            post_probs = {}
            for c in self.classes:
                class_prob = (self.class_counts[c] + self.k) / (self.total_documents + self.k * len(self.classes))
                feature_prob = np.sum((X[i].toarray()[0] + self.k) / (self.feature_counts[c] + self.k * self.vocab_size))
                post_probs[c] = class_prob * feature_prob

            predictions.append(max(post_probs, key=post_probs.get))
        return np.array(predictions)

k_values = [0.25, 0.75, 1]
results = {}

for k in k_values:
    model = NaiveBayes(k=k)
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    results[k] = accuracy
    print(f"Accuracy for k={k}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
