# sentiment_engine.py

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class SentimentEngine:

    def __init__(self, model_path=None, vectorizer_path=None):
        if model_path:
            self.model = pickle.load(open(model_path, 'rb'))
            self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        else:
            self.model = LogisticRegression(max_iter=1000)
            self.vectorizer = TfidfVectorizer(max_features=5000)

        self.labels = {
            1: "Very Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Very Positive"
        }

    def train(self, csv_file):
        df = pd.read_csv(csv_file)
        X = df['review']
        y = df['rating']

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

        pickle.dump(self.model, open("model.pkl", "wb"))
        pickle.dump(self.vectorizer, open("vectorizer.pkl", "wb"))

        print("Training Completed. Model Saved.")

    def predict(self, text):
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        prob = max(self.model.predict_proba(vec)[0])
        
        return {
            "rating": int(pred),
            "sentiment": self.labels[int(pred)],
            "confidence": round(float(prob), 4)
        }
