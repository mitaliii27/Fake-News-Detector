import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "text": [
        "Government announces new education policy",
        "Scientists discover cure for disease",
        "Breaking celebrity caught in fake scandal",
        "Click here to win money instantly fake",
        "NASA launches satellite successfully",
        "Fake miracle cure spreads online",
        "New smartphone improves battery life",
        "False rumor spreads about bank collapse"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Accuracy
pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("model.pkl and vectorizer.pkl saved successfully")
