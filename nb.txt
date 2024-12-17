from collections import defaultdict
import math

# Training data
training_data = [
    ("I love fish. The smoked bass fish was delicious.", "fish"),
    ("The bass fish swam along the line.", "fish"),
    ("He hauled in a big catch of smoked bass fish.", "fish"),
    ("The bass guitar player played a smooth jazz line.", "guitar"),
]

# Test sentence
test_sentence = "He loves jazz. The bass line provided the foundation for the guitar solo in the jazz piece."

# Preprocess text (tokenization)
def tokenize(sentence):
    return sentence.lower().replace('.', '').replace(',', '').split()

# Build vocabulary and calculate probabilities
def train_naive_bayes(training_data):
    word_counts = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)
    vocabulary = set()

    for sentence, label in training_data:
        words = tokenize(sentence)
        class_counts[label] += 1
        for word in words:
            word_counts[label][word] += 1
            vocabulary.add(word)

    total_classes = sum(class_counts.values())
    class_probs = {label: count / total_classes for label, count in class_counts.items()}

    return word_counts, class_probs, vocabulary

# Calculate word probabilities with add-1 smoothing
def calculate_word_probs(word_counts, vocabulary, class_counts):
    word_probs = {}
    vocab_size = len(vocabulary)

    for label, words in word_counts.items():
        total_words = sum(words.values())
        word_probs[label] = {
            word: (count + 1) / (total_words + vocab_size)
            for word, count in words.items()
        }
        word_probs[label]["UNKNOWN"] = 1 / (total_words + vocab_size)

    return word_probs

# Classify a sentence
def classify(sentence, word_probs, class_probs, vocabulary):
    words = tokenize(sentence)
    scores = {}

    for label, class_prob in class_probs.items():
        score = math.log(class_prob)
        for word in words:
            if word in vocabulary:
                score += math.log(word_probs[label].get(word, word_probs[label]["UNKNOWN"]))
            else:
                score += math.log(word_probs[label]["UNKNOWN"])
        scores[label] = score

    return max(scores, key=scores.get)

# Train the Naive Bayes classifier
word_counts, class_probs, vocabulary = train_naive_bayes(training_data)
word_probs = calculate_word_probs(word_counts, vocabulary, class_probs)

# Classify the test sentence
predicted_class = classify(test_sentence, word_probs, class_probs, vocabulary)
print("Predicted sense of 'bass':", predicted_class)
